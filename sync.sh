#!/usr/bin/env bash
# =============================================================================
# sync.sh — synchronise a toupy clone with the cleaned-up remote.
# =============================================================================
#
# Why this exists
# ---------------
# The `dev` branch had its history REWRITTEN upstream (reset onto master's
# lineage), and several branches were deleted on origin.  A plain `git pull` on
# a stale `dev` will fail ("divergent branches" / "unrelated histories") or, if
# forced, merge two unrelated trees.  This script syncs each clone safely:
#
#   * refuses to run with a dirty working tree (unless --stash)
#   * fast-forwards master / twopass (these were NOT rewritten)
#   * for `dev`: fast-forwards if possible; if the local branch has DIVERGED
#     (the history rewrite), it BACKS UP your local dev to a timestamped branch
#     and only then hard-resets it to origin/dev — and only when you pass --yes
#   * prunes deleted remote refs and reports (optionally deletes) orphaned
#     local branches
#
# The new library code (two-pass, FBaP, LocalFSC fix/move, ImageDecorr removed)
# lives on `dev`, NOT on `master` (master is still 0.4.0).  Be on `dev` to use it.
#
# Usage
# -----
#   ./sync.sh                 # safe: fetch, sync master/twopass, sync dev if it
#                             # fast-forwards; if dev diverged, BACK UP + report
#                             # but do NOT reset (dry run for the destructive bit)
#   ./sync.sh --yes           # also perform the dev hard-reset (after backup)
#   ./sync.sh --stash         # stash uncommitted changes first, restore at end
#   ./sync.sh --prune-gone    # also delete local branches whose upstream is gone
#   ./sync.sh --yes --stash --prune-gone   # the full sync in one go
#
# Nothing is ever destroyed without a backup: the pre-reset dev state is kept on
# a local branch `backup/dev-presync-<timestamp>` you can inspect or delete later.
# =============================================================================

set -uo pipefail

REMOTE="origin"
DO_YES=0; DO_STASH=0; DO_PRUNE_GONE=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes)       DO_YES=1 ;;
        --stash)        DO_STASH=1 ;;
        --prune-gone)   DO_PRUNE_GONE=1 ;;
        -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (see --help)"; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; }

# --- sanity -----------------------------------------------------------------
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { err "not a git repo"; exit 1; }
git remote get-url "$REMOTE" >/dev/null 2>&1 || { err "no remote '$REMOTE'"; exit 1; }
cd "$(git rev-parse --show-toplevel)" || exit 1

START_BRANCH=$(git symbolic-ref --quiet --short HEAD || echo "")
STASHED=0

# --- dirty working tree (TRACKED changes only; untracked artifacts are fine) -
if ! git diff --quiet || ! git diff --cached --quiet; then
    if [ "$DO_STASH" = 1 ]; then
        say "Stashing local (tracked) changes"
        git stash push -m "sync.sh $(date -u +%FT%TZ)" && STASHED=1
    else
        err "Working tree has uncommitted changes to tracked files."
        echo "     Commit/push them, or re-run with --stash to set them aside."
        echo "     (Untracked files are ignored — they are not touched by the sync.)"
        git status --short --untracked-files=no | sed 's/^/       /'
        exit 1
    fi
fi

restore() {
    [ -n "$START_BRANCH" ] && git checkout -q "$START_BRANCH" 2>/dev/null
    if [ "$STASHED" = 1 ]; then
        say "Restoring stashed changes"
        git stash pop || warn "stash pop hit a conflict — resolve manually (git stash list)"
    fi
}
trap restore EXIT

# --- fetch + prune ----------------------------------------------------------
say "Fetching $REMOTE and pruning deleted branches"
# --no-tags: this is a BRANCH sync — we don't touch tags.  A clone with local
# tags that differ from origin (older release markers, a past history rewrite)
# would otherwise make 'git fetch --tags' fail with "would clobber existing
# tag" and abort the whole sync.  Tags are left exactly as they are.
git fetch "$REMOTE" --prune --no-tags || { err "fetch failed"; exit 1; }

# --- fast-forward the branches that were NOT rewritten ----------------------
ff_branch() {  # $1 = branch name; only acts if the branch exists locally
    local b="$1"
    git show-ref --verify --quiet "refs/heads/$b" || { warn "no local '$b' (skipping)"; return; }
    git show-ref --verify --quiet "refs/remotes/$REMOTE/$b" || { warn "no $REMOTE/$b (skipping)"; return; }
    git checkout -q "$b" || { warn "could not checkout $b"; return; }
    if git merge --ff-only "$REMOTE/$b" >/dev/null 2>&1; then
        echo "  $b -> fast-forwarded to $REMOTE/$b ($(git rev-parse --short "$b"))"
    else
        warn "$b did NOT fast-forward (you have local commits, or it diverged)."
        warn "  inspect with:  git log --oneline $REMOTE/$b..$b   (unpushed local commits)"
    fi
}
say "Fast-forwarding master / twopass (not rewritten upstream)"
ff_branch master
ff_branch twopass

# --- dev: fast-forward if possible, else backup + (gated) hard reset --------
say "Syncing dev (history was rewritten upstream)"
if ! git show-ref --verify --quiet "refs/heads/dev"; then
    warn "no local 'dev' — creating it to track $REMOTE/dev"
    git branch --track dev "$REMOTE/dev" && echo "  created dev -> $REMOTE/dev"
else
    git checkout -q dev
    LOCAL=$(git rev-parse dev); TARGET=$(git rev-parse "$REMOTE/dev")
    if [ "$LOCAL" = "$TARGET" ]; then
        echo "  dev already up to date ($(git rev-parse --short dev))"
    elif git merge-base --is-ancestor dev "$REMOTE/dev"; then
        git merge --ff-only "$REMOTE/dev" >/dev/null && echo "  dev fast-forwarded (no rewrite on this clone)"
    else
        BACKUP="backup/dev-presync-$(date +%Y%m%d-%H%M%S)"
        git branch "$BACKUP" dev
        UNPUSHED=$(git rev-list --count "$REMOTE/dev..dev")
        warn "Local dev has DIVERGED from $REMOTE/dev (the upstream history rewrite)."
        echo "       Backed up your current dev to:  $BACKUP  ($UNPUSHED commit(s) not on $REMOTE/dev)"
        if [ "$DO_YES" = 1 ]; then
            git reset --hard "$REMOTE/dev"
            echo "  dev hard-reset to $REMOTE/dev ($(git rev-parse --short dev))."
            echo "       Old state preserved on $BACKUP — delete it once you're satisfied:"
            echo "         git branch -D $BACKUP"
        else
            warn "NOT resetting (dry run). Re-run with --yes to hard-reset dev to $REMOTE/dev."
            echo "       (Your work is safe on $BACKUP either way.)"
        fi
    fi
fi

# --- orphaned local branches (upstream deleted) -----------------------------
say "Local branches whose upstream branch was deleted on $REMOTE"
GONE=$(git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads \
        | awk '$2=="[gone]"{print $1}')
if [ -z "$GONE" ]; then
    echo "  none"
else
    echo "$GONE" | sed 's/^/  gone: /'
    if [ "$DO_PRUNE_GONE" = 1 ]; then
        for b in $GONE; do
            [ "$b" = "$START_BRANCH" ] && { warn "skipping current branch $b"; continue; }
            git branch -D "$b" && echo "  deleted $b"
        done
    else
        echo "  (re-run with --prune-gone to delete these)"
    fi
fi

# --- summary ----------------------------------------------------------------
say "Done"
echo "  dev    = $(git rev-parse --short "$REMOTE/dev")   <- new library (two-pass, FBaP, ...)"
echo "  master = $(git rev-parse --short "$REMOTE/master")   <- still 0.4.0 (unchanged)"
echo
echo "  If toupy was installed with 'pip install .' (not editable), reinstall so the"
echo "  new toupy.tomo modules are importable:   pip install -e .   (from this dir)"
