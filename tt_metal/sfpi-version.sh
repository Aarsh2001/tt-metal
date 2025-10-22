# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.4.0-hll
sfpi_x86_64_deb_md5=c9cf6321c3141330e4a46071c6faa9c8
sfpi_x86_64_rpm_md5=afe184a325616d2ebca15fc7a6ad15b0
sfpi_x86_64_txz_md5=e8735bc66fd3bc9c6cf2114fae62cc2f
