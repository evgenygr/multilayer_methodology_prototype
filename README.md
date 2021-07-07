# Multilayer methodology prototype

This repo contains a prototype of a formalized methodology for exploratory wells placement. It was developed in the frameworks of a project initiated by Science and Technology center of Gazpromneft (the 2nd largest russian oil-and-gas company). The problem that this prototype addresses is the following: what is the best position of next exploratory well from the standpoint of maximization of awareness about the inner structure of the hydrocarbon-bearing formation given the data from wells that are already present on a field. To check out the performance of the prototype one needs to:

1. create virtual environment
2. install packages from requirements.txt into venv
3. open terminal and activate venv
4. to run the experiment, run the following command: `python run_multilayer_field_data_experiment.py -s settings_multilayer_field_data_experiment.json -f OilCase2020.csv -r OilCase2020_field_run_results --verbose --concise`
5. to visualize actions taken by geoBot, run: `python visualize_actions.py -d OilCase2020_field_run_results -o OilCase2020_figures -t ntg -m`
6. to plot a decline profile of target value RMSE, run: `python visualise_rmse_decline_profiles.py -d OilCase2020_field_run_results/RMSE_decline_profile.csv  -o RMSE_decline_profiles/summary.png`

Python version is >=3.7

## The use of pandas

Almost all files in `src` use some standard feature set of pandas, but the most extensive use of Pandas' capabilities can be seen in `src/composite_metrics.py`.
