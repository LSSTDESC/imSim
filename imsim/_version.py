
# Conventions:
#   M.m means a development version not ready for prime time yet.  Not stable
#   M.m.r is a regular release:
#      Intended to be API stable with previous M.x.y releases.
#      New features may be added with the major (M) series, but the intent is to not
#      break public API-based code from previous versions in the series.
#      Updates that change the revision number (r) should only have bug fixes.
#   M.m.r-* is some kind of prerelease or release candidate for alpha or beta testing.
#
# cf. https://semver.org/ for more info

# This should be updated before a release.
__version__ = '2.0'

# This will work if the version has end tags like 2.0.0-rc.1
__version_info__ = tuple(map(lambda x:int(x.split('-')[0]), __version__.split('.')))[:3]
