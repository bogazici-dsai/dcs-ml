    Notes
  
	•	Tested with Python 3.9.
 
	•	Create a conda environment named llmrl.
 
	•	After cloning the repository, install all dependencies using requirements.txt.
 
	•	Make sure Ollama is installed on your system.
 
	•	Use the LLaMA 3.1 8B model locally via Ollama.
 
	•	SUMO should be correctly installed and accessible from the environment.
                eg:  os.environ['SUMO_HOME'] = '/opt/miniconda3/envs/llmrl'
	              os.environ['PATH'] = f"/opt/miniconda3/envs/llmrl/bin:{os.environ['PATH']}"
 
	•	You can run the simulation with a single command once setup is complete (rl_llm_tsc.py).
