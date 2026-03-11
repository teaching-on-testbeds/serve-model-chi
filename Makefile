PROFILE ?= amd

INDEX_OUT := index.md
INDEX_NVIDIA_OUT := index_nvidia.md
INDEX_AMD_OUT := index_amd.md

.PHONY: all build clean

all: build

build: $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) $(INDEX_OUT) 0_intro_nvidia.ipynb 0_intro_amd.ipynb 0_intro.ipynb 1_create_lease_nvidia.ipynb 1_create_lease_amd.ipynb 1_create_lease.ipynb 2_create_server_nvidia.ipynb 2_create_server_amd.ipynb 2_create_server.ipynb 3_prepare_data.ipynb 4_launch_jupyter.ipynb workspace/5_measure_torch.ipynb workspace/6_measure_onnx.ipynb workspace/7_optimize_onnx.ipynb workspace/8_ep_onnx_nvidia.ipynb workspace/8_ep_onnx_amd.ipynb workspace/8_ep_onnx.ipynb

clean:
	rm -f $(INDEX_OUT) $(INDEX_NVIDIA_OUT) $(INDEX_AMD_OUT) \
		0_intro_nvidia.ipynb \
		0_intro_amd.ipynb \
		0_intro.ipynb \
		1_create_lease_nvidia.ipynb \
		1_create_lease_amd.ipynb \
		1_create_lease.ipynb \
		2_create_server_nvidia.ipynb \
		2_create_server_amd.ipynb \
		2_create_server.ipynb \
		3_prepare_data.ipynb \
		4_launch_jupyter.ipynb \
		workspace/5_measure_torch.ipynb \
		workspace/6_measure_onnx.ipynb \
		workspace/7_optimize_onnx.ipynb \
		workspace/8_ep_onnx_nvidia.ipynb \
		workspace/8_ep_onnx_amd.ipynb \
		workspace/8_ep_onnx.ipynb

$(INDEX_NVIDIA_OUT): snippets/intro_nvidia.md snippets/create_lease_nvidia.md snippets/create_server_nvidia.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx_nvidia.md snippets/footer.md
	cat snippets/intro_nvidia.md \
		snippets/create_lease_nvidia.md \
		snippets/create_server_nvidia.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx_nvidia.md \
		> index.nvidia.tmp.md
	grep -v '^:::' index.nvidia.tmp.md > $(INDEX_NVIDIA_OUT)
	rm index.nvidia.tmp.md
	cat snippets/footer.md >> $(INDEX_NVIDIA_OUT)

$(INDEX_AMD_OUT): snippets/intro_amd.md snippets/create_lease_amd.md snippets/create_server_amd.md snippets/prepare_data.md snippets/launch_jupyter.md snippets/measure_torch.md snippets/measure_onnx.md snippets/optimize_onnx.md snippets/ep_onnx_amd.md snippets/footer.md
	cat snippets/intro_amd.md \
		snippets/create_lease_amd.md \
		snippets/create_server_amd.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx_amd.md \
		> index.amd.tmp.md
	grep -v '^:::' index.amd.tmp.md > $(INDEX_AMD_OUT)
	rm index.amd.tmp.md
	cat snippets/footer.md >> $(INDEX_AMD_OUT)

$(INDEX_OUT): index_$(PROFILE).md
	cp index_$(PROFILE).md $(INDEX_OUT)

0_intro_nvidia.ipynb: snippets/intro_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/intro_nvidia.md \
				-o 0_intro_nvidia.ipynb
	sed -i 's/attachment://g' 0_intro_nvidia.ipynb

0_intro_amd.ipynb: snippets/intro_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/intro_amd.md \
				-o 0_intro_amd.ipynb
	sed -i 's/attachment://g' 0_intro_amd.ipynb

0_intro.ipynb: 0_intro_$(PROFILE).ipynb
	cp 0_intro_$(PROFILE).ipynb 0_intro.ipynb

1_create_lease_nvidia.ipynb: snippets/create_lease_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_lease_nvidia.md \
				-o 1_create_lease_nvidia.ipynb
	sed -i 's/attachment://g' 1_create_lease_nvidia.ipynb

1_create_lease_amd.ipynb: snippets/create_lease_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_lease_amd.md \
				-o 1_create_lease_amd.ipynb
	sed -i 's/attachment://g' 1_create_lease_amd.ipynb

1_create_lease.ipynb: 1_create_lease_$(PROFILE).ipynb
	cp 1_create_lease_$(PROFILE).ipynb 1_create_lease.ipynb

2_create_server_nvidia.ipynb: snippets/create_server_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_server_nvidia.md \
				-o 2_create_server_nvidia.ipynb
	sed -i 's/attachment://g' 2_create_server_nvidia.ipynb

2_create_server_amd.ipynb: snippets/create_server_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_server_amd.md \
				-o 2_create_server_amd.ipynb
	sed -i 's/attachment://g' 2_create_server_amd.ipynb

2_create_server.ipynb: 2_create_server_$(PROFILE).ipynb
	cp 2_create_server_$(PROFILE).ipynb 2_create_server.ipynb

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

workspace/8_ep_onnx_nvidia.ipynb: snippets/ep_onnx_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/ep_onnx_nvidia.md \
				-o workspace/8_ep_onnx_nvidia.ipynb
	sed -i 's/attachment://g' workspace/8_ep_onnx_nvidia.ipynb

workspace/8_ep_onnx_amd.ipynb: snippets/ep_onnx_amd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/ep_onnx_amd.md \
				-o workspace/8_ep_onnx_amd.ipynb
	sed -i 's/attachment://g' workspace/8_ep_onnx_amd.ipynb

workspace/8_ep_onnx.ipynb: workspace/8_ep_onnx_$(PROFILE).ipynb
	cp workspace/8_ep_onnx_$(PROFILE).ipynb workspace/8_ep_onnx.ipynb
