"""Cantera adapter layer (cantera_tool)

Except for this file, sources in this directory primarily originate from the
Cantera project (https://cantera.org, GitHub: https://github.com/Cantera/cantera).
To make them easier to use within this project, we applied a few minor, environment-
specific adjustments (e.g., path/import tweaks, small interface refinements, and
occasional type hints). Core algorithms and behavior remain consistent with upstream.

License and compliance
- Upstream Cantera is distributed under the BSD-3-Clause license. In practice:
	- You may copy, modify, and redistribute provided you preserve the original
		copyright and license notices.
	- You must not imply endorsement by the upstream authors or copyright holders.
	- You must include the BSD-3-Clause license text with your distributions.
- Third-party source files in this folder remain covered by BSD-3-Clause; newly
	added adaptation code follows the LICENSE at the project root. These scopes
	coexist without conflict.

Compliance tips (for distribution/open-source):
1) Ship the upstream license text and copyright notices (e.g., in release
	 notes or a THIRD_PARTY_NOTICES file), and reference Cantera with links.
2) In modified files or change logs, note "modified from Cantera" and summarize
	 the types of changes (line-by-line markings are not necessary).
3) If you plan to track upstream, record the upstream version/commit for traceability.

Notes
- This __init__ serves as an informational entry point and does not export
	symbols; if a unified API is needed later, exports can be aggregated here.
"""

__all__ = []

