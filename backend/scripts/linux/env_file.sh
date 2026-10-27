#!/usr/bin/env bash
# Shared by setup_env.sh and install_strata.sh; sourced, never run. Needs PROJECT_ROOT set.
# LOG_TAG is the prefix of the lines it prints ([setup_env] unless the caller sets another).

# Record what setup_env actually resolved, so the service reads a decision instead of
# repeating the search. PATH is per-process: setup runs in your shell, the service may run
# under a service manager with a different one, and then the two disagree. The file ends up with
# exactly one line per key: an override that resolves is kept as it is, one that
# points at nothing is replaced by what was actually verified, and duplicates are collapsed.
record_env_setting() {
    local key="$1" resolved="$2" env_file="$PROJECT_ROOT/.env"
    [[ -n "$resolved" ]] || return 0
    if [[ ! -f "$env_file" ]]; then
        # settings.py falls back to .env.example when .env is missing; creating one here would
        # change which file the service loads, so only say what could not be recorded.
        echo "${LOG_TAG:-[setup_env]} no .env, so $key was not recorded (resolved: $resolved)"
        return 0
    fi
    # Exactly one line for the key survives this. Appending whenever no *live* line
    # matched left .env.example's commented example sitting above a second entry, so the file
    # showed the key twice; and once two live lines exist python-dotenv silently keeps the last,
    # which is not the one someone editing the first would expect.
    local live="^[[:space:]]*$key[[:space:]]*="
    local commented="^[[:space:]]*#[[:space:]]*$key[[:space:]]*="
    local stripped previous dropped total bom
    stripped="$(mktemp)"
    # Drop a leading BOM first: it hides a first-line key from every ^ anchor below, and
    # python-dotenv would read it as part of that key's name.
    bom="$(printf '\357\273\277')"
    sed "1s/^$bom//" "$env_file" > "$stripped"

    total="$(grep -cE "$live" "$stripped" || true)"
    dropped=$(( total > 0 ? total - 1 : 0 ))
    previous="$(grep -m1 -E "$live" "$stripped" 2>/dev/null |
        sed -E "s/$live[[:space:]]*//" | tr -d '"'"'"'' | sed 's/[[:space:]]*$//' || true)"

    local rewritten
    rewritten="$(mktemp)"
    awk -v setting="$key=$resolved" -v live="$live" -v commented="$commented" '
        {
            if ($0 ~ live) {
                if (placed) { next }        # a duplicate: drop it
                buf[++n] = setting
                placed = 1
                next
            }
            buf[++n] = $0
        }
        END {
            if (!placed) {
                # Take over the commented example in place, so the key stays where the template
                # put it instead of the file growing a second block saying the same thing.
                for (i = 1; i <= n; i++) {
                    if (buf[i] ~ commented) { buf[i] = setting; placed = 1; break }
                }
            }
            if (!placed) {
                buf[++n] = ""
                buf[++n] = "# Recorded by setup: a path it resolved, so the service does not have to"
                buf[++n] = "# re-resolve it from a possibly different PATH."
                buf[++n] = setting
            }
            for (i = 1; i <= n; i++) print buf[i]
        }
    ' "$stripped" > "$rewritten"

    # cat rather than mv: keeps the original inode, owner and mode, which matter for a .env.
    cat "$rewritten" > "$env_file"
    rm -f "$stripped" "$rewritten"

    if [[ "$previous" == "$resolved" ]]; then
        echo "${LOG_TAG:-[setup_env]} .env already records $key=$resolved"
    elif [[ -n "$previous" ]]; then
        echo "${LOG_TAG:-[setup_env]} .env $key updated to $resolved (was $previous)"
    else
        echo "${LOG_TAG:-[setup_env]} recorded $key=$resolved in .env"
    fi
    (( dropped > 0 )) &&
        echo "${LOG_TAG:-[setup_env]} removed $dropped duplicate $key line(s) from .env"
    return 0
}
