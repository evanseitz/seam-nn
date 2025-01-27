Optimized Logomaker Implementation
================================

This is an optimized version of the original Logomaker package (https://github.com/jbkinney/logomaker) by Ammar Tareen and Justin Kinney. The optimization focuses on batch processing capabilities, significantly improving performance for generating multiple sequence logos.

Key Improvements:
- Efficient batch processing of multiple logos
- Path caching for improved performance
- Minimized object creation
- Optimized transformation operations

Performance Gains:
Original implementation: ~2 seconds per logo (249 x 4 matrix)
Optimized batch implementation: ~0.013 seconds per logo (processes 1000 logos in 13 seconds)

The BatchLogo class provides these optimizations while maintaining compatibility with the original Logomaker functionality. For detailed implementation notes, please see the comments at the bottom of BatchLogo.py.

Credits:
- Original Logomaker: Ammar Tareen and Justin Kinney, 2019-2024
- Batch processing optimization: Evan Seitz, 2025
- Additional optimizations: Implemented with Claude AI assistance

For the original package and documentation, visit: https://github.com/jbkinney/logomaker