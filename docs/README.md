# Documentation Index

This directory contains all documentation for the kmeans-seeding project, organized by audience and purpose.

## For Users

Start here if you're using the kmeans-seeding package:

- **[README.md](../README.md)** - Main package documentation with installation and usage examples
- **[publishing/QUICK_START_GUIDE.md](publishing/QUICK_START_GUIDE.md)** - Quick start guide for getting started

## For Developers

Read these if you're contributing to the codebase:

- **[CLAUDE.md](../CLAUDE.md)** - Comprehensive guide for AI assistants (covers architecture, build system, workflows)
- **[development/SETUP_SUMMARY.md](development/SETUP_SUMMARY.md)** - Development environment setup
- **[development/CORRECT_ARCHITECTURE.md](development/CORRECT_ARCHITECTURE.md)** - Project architecture overview
- **[development/RESTRUCTURE_PLAN.md](development/RESTRUCTURE_PLAN.md)** - Historical restructuring documentation

### Key Development Resources

- **Tests**: `../tests/` - Full test suite with pytest
- **C++ Source**: `../cpp/src/` - Core algorithms implementation
- **Python API**: `../python/kmeans_seeding/` - User-facing Python interface
- **Build Config**: `../pyproject.toml`, `../setup.py`, `../cpp/CMakeLists.txt`

## For Maintainers

Essential guides for package maintainers and release managers:

- **[publishing/PUBLICATION_GUIDE.md](publishing/PUBLICATION_GUIDE.md)** - Complete PyPI publishing guide
- **[publishing/PUBLISH_COMMANDS.md](publishing/PUBLISH_COMMANDS.md)** - Release commands reference
- **[publishing/PRE_PUBLISH_CHECKLIST.md](publishing/PRE_PUBLISH_CHECKLIST.md)** - Pre-release checklist

### Quick Publishing Commands

```bash
# Update version in python/kmeans_seeding/__init__.py and pyproject.toml
# Then run:
./republish.sh  # Automated build and publish
```

## Archive

Historical documentation from the development process:

- **[archive/](archive/)** - Contains process documentation, fixes, and status updates from development
  - `BEFORE_AFTER_FIX.md` - Documentation of PyPI packaging fixes
  - `FIX_PYPI_PACKAGE.md` - PyPI package troubleshooting
  - `URGENT_FIX_SUMMARY.md` - Critical fix summaries
  - `PROJECT_STATUS.md` - Historical project status
  - `NEXT_STEPS.md` - Historical roadmap
  - `PUBLISH_NOW.md` - Publishing notes
  - `READY_FOR_PUBLICATION.md` - Publication readiness checklist
  - `TESTS_COMPLETE.md` - Test completion documentation

These files are kept for historical reference but may be outdated.

## Additional Resources

- **Research**: `../quantization_analysis/` - Empirical quantization dimension analysis (separate from main package)
- **Legacy Code**: `../archive/rs_kmeans/`, `../archive/fast_k_means_2020/` - Historical implementations
- **Benchmarks**: `../benchmarks/` - Performance benchmarking scripts
- **Examples**: `../examples/` - Usage examples

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── development/           # Developer documentation
│   ├── SETUP_SUMMARY.md
│   ├── CORRECT_ARCHITECTURE.md
│   └── RESTRUCTURE_PLAN.md
├── publishing/            # Maintainer/release documentation
│   ├── QUICK_START_GUIDE.md
│   ├── PUBLICATION_GUIDE.md
│   ├── PUBLISH_COMMANDS.md
│   └── PRE_PUBLISH_CHECKLIST.md
└── archive/               # Historical process documentation
    ├── BEFORE_AFTER_FIX.md
    ├── FIX_PYPI_PACKAGE.md
    ├── URGENT_FIX_SUMMARY.md
    ├── PROJECT_STATUS.md
    ├── NEXT_STEPS.md
    ├── PUBLISH_NOW.md
    ├── READY_FOR_PUBLICATION.md
    └── TESTS_COMPLETE.md
```

## Getting Help

- **Issues**: https://github.com/pcshah2004/kmeans-seeding/issues
- **Discussions**: https://github.com/pcshah2004/kmeans-seeding/discussions
- **Email**: cs1221594@cse.iitd.ac.in
