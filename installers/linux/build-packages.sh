#!/usr/bin/env bash
# Build RAI's .deb and .rpm from an unpacked Linux release folder.
#
#   installers/linux/build-packages.sh <unpacked-release-dir> <version> <out-dir>
#
# Both packages lay RAI out the same way:
#   /usr/bin/rai                         the program
#   /usr/bin/rai-studio, rai-get-model   links into /usr/lib/rai
#   /usr/lib/rai/                        the launcher, the model fetcher, the model list
#   /usr/share/applications/             a "RAI Studio" menu entry
#   /usr/share/doc/rai/                  README, licence and notices
# Needs dpkg-deb, and rpmbuild for the .rpm (on Ubuntu: apt install rpm).
set -euo pipefail

src="${1:?unpacked release folder}"
version="${2:?version}"
out="${3:?output folder}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

root="$work/root"
install -Dm755 "$src/rai" "$root/usr/bin/rai"
install -Dm755 "$src/launchers/rai-studio.sh" "$root/usr/lib/rai/rai-studio.sh"
install -Dm755 "$src/launchers/rai-get-model" "$root/usr/lib/rai/rai-get-model"
install -Dm644 "$src/launchers/models.tsv" "$root/usr/lib/rai/models.tsv"
ln -s ../lib/rai/rai-studio.sh "$root/usr/bin/rai-studio"
ln -s ../lib/rai/rai-get-model "$root/usr/bin/rai-get-model"
for doc in README.md NOTICE THIRD-PARTY-NOTICES.md; do
    install -Dm644 "$src/$doc" "$root/usr/share/doc/rai/$doc"
done
install -Dm644 "$src/LICENSE" "$root/usr/share/doc/rai/copyright"

# The menu entry starts the launcher on PATH; Terminal=true because the window
# is where the server's progress and any download question appear.
install -d "$root/usr/share/applications"
cat > "$root/usr/share/applications/rai-studio.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Version=1.0
Name=RAI Studio
GenericName=Local language model chat
Comment=Chat with a language model on this computer, in your browser
Exec=rai-studio
Icon=utilities-terminal
Terminal=true
Categories=Development;Science;Utility;
Keywords=llm;ai;chat;offline;
StartupNotify=false
DESKTOP

size_kb=$(du -sk "$root" | cut -f1)
summary="Run language models on your computer's processor"
read -r -d '' long <<'LONG' || true
RAI runs language models on an ordinary processor, with no graphics card
and no Python. Download a model with rai-get-model, then chat with it in
your browser with rai-studio, or use rai run and rai serve directly.
LONG

# --- .deb -----------------------------------------------------------------------
deb="$work/deb"
cp -a "$root" "$deb"
install -d "$deb/DEBIAN"
{
    printf 'Package: rai\n'
    printf 'Version: %s\n' "$version"
    printf 'Section: utils\n'
    printf 'Priority: optional\n'
    printf 'Architecture: amd64\n'
    printf 'Maintainer: ClassEve <security@classeve.com>\n'
    printf 'Installed-Size: %s\n' "$size_kb"
    # glibc 2.28 is the floor the release is built for; rai-get-model reads
    # its model list with awk and downloads with curl. xdg-utils only opens a
    # browser, which a desktop already has and a server does not need.
    printf 'Depends: libc6 (>= 2.28), awk\n'
    printf 'Recommends: curl\n'
    printf 'Suggests: xdg-utils\n'
    printf 'Homepage: https://github.com/Classevelabs/rai\n'
    printf 'Description: %s\n' "$summary"
    printf '%s\n' "$long" | sed 's/^/ /'
} > "$deb/DEBIAN/control"
dpkg-deb --root-owner-group --build "$deb" "$out/rai_${version}_amd64.deb" >/dev/null
echo "built $out/rai_${version}_amd64.deb"

# --- .rpm -----------------------------------------------------------------------
if command -v rpmbuild >/dev/null 2>&1; then
    top="$work/rpm"
    mkdir -p "$top"/{BUILD,RPMS,SOURCES,SPECS,SRPMS,BUILDROOT}
    cat > "$top/SPECS/rai.spec" <<SPEC
Name:           rai
Version:        ${version}
Release:        1
Summary:        ${summary}
License:        Apache-2.0
URL:            https://github.com/Classevelabs/rai
# rai links glibc and nothing else, built for 2.28 and newer, so the one
# requirement is stated rather than scanned.
AutoReqProv:    no
Requires:       glibc >= 2.28
# rai-get-model reads its model list with awk; a minimal openSUSE has none.
# gawk is the package every rpm distribution names it.
Requires:       gawk
Recommends:     curl

%description
${long}

%install
mkdir -p %{buildroot}
cp -a ${root}/. %{buildroot}/

%files
/usr/bin/rai
/usr/bin/rai-studio
/usr/bin/rai-get-model
/usr/lib/rai
/usr/share/applications/rai-studio.desktop
%doc /usr/share/doc/rai
SPEC
    rpmbuild --quiet -bb --define "_topdir $top" --define "_build_id_links none" \
        --define "__os_install_post %{nil}" "$top/SPECS/rai.spec"
    cp "$top"/RPMS/x86_64/rai-"${version}"-1.x86_64.rpm "$out/"
    echo "built $out/rai-${version}-1.x86_64.rpm"
else
    echo "rpmbuild is not installed; no .rpm" >&2
    exit 1
fi
