INDEX_OUT := index.md
INDEX_AMD_OUT := index_amd.md
INDEX_INTEL_OUT := index_intel.md

.PHONY: all build clean

all: build

build: $(INDEX_AMD_OUT) $(INDEX_INTEL_OUT) $(INDEX_OUT) 0_intro_amd.ipynb 0_intro_intel.ipynb 1_create_lease_amd.ipynb 1_create_lease_intel.ipynb 2_create_server.ipynb 3_prepare_data.ipynb 4_launch_jupyter.ipynb workspace/5_measure_torch.ipynb workspace/6_measure_onnx.ipynb workspace/7_optimize_onnx.ipynb workspace/8_ep_onnx.ipynb

clean:
	rm -f $(INDEX_OUT) $(INDEX_AMD_OUT) $(INDEX_INTEL_OUT) \
		0_intro_amd.ipynb \
		0_intro_intel.ipynb \
		1_create_lease_amd.ipynb \
		1_create_lease_intel.ipynb \
		2_create_server.ipynb \
		3_prepare_data.ipynb \
		4_launch_jupyter.ipynb \
		workspace/5_measure_torch.ipynb \
		workspace/6_measure_onnx.ipynb \
		workspace/7_optimize_onnx.ipynb \
		workspace/8_ep_onnx.ipynb

$(INDEX_AMD_OUT): snippets/intro_amd.md snippets/create_lease_amd.md snippets/create_server.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx.md snippets/footer.md
	cat snippets/intro_amd.md \
		snippets/create_lease_amd.md \
		snippets/create_server.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx.md \
		> index.amd.tmp.md
	grep -v '^:::' index.amd.tmp.md > $(INDEX_AMD_OUT)
	rm index.amd.tmp.md
	cat snippets/footer.md >> $(INDEX_AMD_OUT)

$(INDEX_INTEL_OUT): snippets/intro_intel.md snippets/create_lease_intel.md snippets/create_server.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx.md snippets/footer.md
	cat snippets/intro_intel.md \
		snippets/create_lease_intel.md \
		snippets/create_server.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx.md \
		> index.intel.tmp.md
	grep -v '^:::' index.intel.tmp.md > $(INDEX_INTEL_OUT)
	rm index.intel.tmp.md
	cat snippets/footer.md >> $(INDEX_INTEL_OUT)

$(INDEX_OUT): $(INDEX_AMD_OUT)
	cp $(INDEX_AMD_OUT) $(INDEX_OUT)

0_intro_amd.ipynb: snippets/intro_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/intro_amd.md \
				-o 0_intro_amd.ipynb
	sed -i 's/attachment://g' 0_intro_amd.ipynb

0_intro_intel.ipynb: snippets/intro_intel.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/intro_intel.md \
				-o 0_intro_intel.ipynb
	sed -i 's/attachment://g' 0_intro_intel.ipynb

1_create_lease_amd.ipynb: snippets/create_lease_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_lease_amd.md \
				-o 1_create_lease_amd.ipynb
	sed -i 's/attachment://g' 1_create_lease_amd.ipynb

1_create_lease_intel.ipynb: snippets/create_lease_intel.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_lease_intel.md \
				-o 1_create_lease_intel.ipynb
	sed -i 's/attachment://g' 1_create_lease_intel.ipynb

2_create_server.ipynb: snippets/create_server.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_server.md \
				-o 2_create_server.ipynb
	sed -i 's/attachment://g' 2_create_server.ipynb

3_prepare_data.ipynb: snippets/prepare_data.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/prepare_data.md \
				-o 3_prepare_data.ipynb
	sed -i 's/attachment://g' 3_prepare_data.ipynb

4_launch_jupyter.ipynb: snippets/launch_jupyter.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/launch_jupyter.md \
				-o 4_launch_jupyter.ipynb
	sed -i 's/attachment://g' 4_launch_jupyter.ipynb

workspace/5_measure_torch.ipynb: snippets/measure_torch.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/measure_torch.md \
				-o workspace/5_measure_torch.ipynb
	sed -i 's/attachment://g' workspace/5_measure_torch.ipynb

workspace/6_measure_onnx.ipynb: snippets/measure_onnx.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/measure_onnx.md \
				-o workspace/6_measure_onnx.ipynb
	sed -i 's/attachment://g' workspace/6_measure_onnx.ipynb

workspace/7_optimize_onnx.ipynb: snippets/optimize_onnx.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/optimize_onnx.md \
				-o workspace/7_optimize_onnx.ipynb
	sed -i 's/attachment://g' workspace/7_optimize_onnx.ipynb

workspace/8_ep_onnx.ipynb: snippets/ep_onnx.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/ep_onnx.md \
				-o workspace/8_ep_onnx.ipynb
	sed -i 's/attachment://g' workspace/8_ep_onnx.ipynb
