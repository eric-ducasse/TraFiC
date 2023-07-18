# Recommended installation of Python 3 for <tt>TraFiC</tt>
Python 3.10 with modules <tt><b>numpy</b></tt>, <tt><b>scipy</b></tt>, <tt><b>matplotlib</b></tt>, <tt><b>PyQt5</b></tt> and <tt><b>imageio</b></tt>.  
## To install a specific virtual environment:
1. Download <b>Miniconda</b> here: [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html), making sure you select the correct version for your operating system, and that the installation path contains no spaces, accented characters, cedillas or hyphens.
2. Launch <b>Anaconda Prompt (Miniconda3)</b> in MS Windows, or a <b>Terminal</b> in Linux, Mac OSX, M1 or M2.
3. Create a virtual environment named <tt><b>trafic_env</b></tt> by:
    > <tt><b>(base)...> conda create -n trafic_env python=3.10 -y </b></tt>   
4. Enter this environment with the following command:
    > <tt><b>(base)...> conda activate trafic_env</b></tt>  
5. Such that the prompt becomes:  
   > <tt><b>(trafic_env)...></b></tt>
6. Module installation:
   > <tt><b>(trafic_env)...> conda install numpy -y </b></tt>  
   > <tt><b>(trafic_env)...> conda install scipy -y </b></tt>  
   > <tt><b>(trafic_env)...> conda install matplotlib -y </b></tt>   
   > <tt><b>(trafic_env)...> conda install imageio -y </b></tt>   
7. [optional] IDE installation:
   > <tt><b>(trafic_env)...> pip install idlex </b></tt>   
   > <tt><b>(trafic_env)...> conda install jupyter -y</b></tt>   
   > <tt><b>(trafic_env)...> conda install spyder -y</b></tt>   
8. To run <tt><b>my_code.py</b></tt> located in <tt><b>my_folder</b></tt>:
   > <tt><b>(base)...> conda activate trafic_env</b></tt> </b></tt>   
   > <tt><b>(trafic_env)...> cd <i><my_folder path></i> </b></tt>  
   > <tt><b>(trafic_env)...> python my_code.py </b></tt>  
9. To launch <b>Idlex</b>:
   > <tt><b>(base)...> conda activate trafic_env</b></tt> 
   - Mac/Linux: 
   > <tt><b>(trafic_env)...> idlex </b></tt>  
   - Windows:  
   > <tt><b>(trafic_env)...> cd <i><Miniconda3_path> </i> \envs\trafic_env</b></tt>  
   > <tt><b>(trafic_env)...> python Scripts\idlex.py </b></tt>  
10. To launch <b>Jupyter notebook</b>:
   > <tt><b>(base)...> conda activate trafic_env</b></tt>   
   > <tt><b>(trafic_env)...> jupyter notebook <i><folder_path></i> </b></tt>   
11. To launch <b>Spyder</b>:
   > <tt><b>(base)...> conda activate trafic_env</b></tt>   
   > <tt><b>(trafic_env)...> spyder </b></tt>   