# Copyright 2026 BeingQ-space
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Entry point: launches the DocuRAG web UI."""

import os
import shutil
from pathlib import Path

from docurag import pipeline as rag
from docurag.ui import build_ui


def seed_demo_docs():
	"""Copies the committed demo PDFs into the index path and ingests them."""

	demo = Path("demo_docs")
	if not demo.exists():
		return
	
	rag.ensure_paths()
	for pdf in demo.glob("*.pdf"):
		dest = rag.PDF_PATH / pdf.name
		if not dest.exists():
			shutil.copy(pdf, dest)
	
	rag.add_new_pdfs_if_needed()

if __name__ == "__main__":
	if os.environ.get("DEMO_MODE"):
		seed_demo_docs()
	build_ui().launch()
