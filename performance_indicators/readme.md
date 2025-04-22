Configuration is in mmx_kpis.toml: Paths to data, where to save the results, which indicators to compute

-ex: name of the folder with processed data  (after running strategic pipeline)

-c: if we want to compare 2 experiments (need to have computed the indicators first for each experiment individually)

-ppv: post-processing version (default 0): defines the number that is in file names, e.g. possible_itineraries_1.csv

The results are saved into a specified folder 'indicators' (path in .toml) as csv or plots.
	# Examples of usage
 
	python3 mmx_kpis.py -ex processed_cs10.pp00.so00_c1
 
	python3 mmx_kpis.py -c processed_cs10.pp00.so00_c1 processed_cs10.pp10.so00_c1
 
	python3 mmx_kpis.py -c processed_cs10.pp00.so00_c2 processed_c1_replan -ppv 0 1
