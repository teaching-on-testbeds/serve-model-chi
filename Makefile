all: \
	index.md \
	0_intro.ipynb \
	1_create_lease.ipynb \
	2_create_server.ipynb \
	3_prepare_data.ipynb \
	4_launch_jupyter.ipynb \
	workspace/5_measure_torch.ipynb \
	workspace/6_measure_onnx.ipynb \
	workspace/7_optimize_onnx.ipynb \
	workspace/8_ep_onnx.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_lease.ipynb \
	2_create_server.ipynb \
	3_prepare_data.ipynb \
	4_launch_jupyter.ipynb \
	workspace/5_measure_torch.ipynb \
	workspace/6_measure_onnx.ipynb \
	workspace/7_optimize_onnx.ipynb \
	workspace/8_ep_onnx.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_lease.md \
		snippets/create_server.md \
		snippets/prepare_data.md \
		snippets/launch_jupyter.md \
		snippets/measure_torch.md \
		snippets/measure_onnx.md \
		snippets/optimize_onnx.md \
		snippets/ep_onnx.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_lease.ipynb: snippets/create_lease.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/create_lease.md \
				-o 1_create_lease.ipynb  
	sed -i 's/attachment://g' 1_create_lease.ipynb


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
