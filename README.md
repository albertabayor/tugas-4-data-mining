# Run Jupyter Notebook in VS Code
This project contains Jupyter Notebooks that can be run and edited using Visual Studio Code (VS Code). Follow the instructions below to set up your environment and start working with the notebooks.
## Prerequisites
- Install [Visual Studio Code](https://code.visualstudio.com/).
- Install the [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
- Install [Jupyter extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
- Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
## Setting Up the Environment
1. Clone this repository to your local machine.
2. Open the cloned repository in VS Code.
3. (Optional) Create a virtual environment for the project:
   ```bash
   python -m venv ./venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Running the Notebooks
1. Open the desired `.ipynb` file in VS Code.
2. You can run individual cells by clicking the "Run" button that appears when you hover over a cell or by using the keyboard shortcut `Shift + Enter`.
3. You can also run all cells by clicking on the "Run All" button at the top of the notebook interface.
## Additional Tips
- Use the command palette (`Ctrl + Shift + P` or `Cmd + Shift + P` on macOS) to access various Jupyter commands.
- You can change the kernel by clicking on the kernel name in the top right corner of the notebook interface.
- For more information, refer to the [official VS Code documentation for Jupyter Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

By following these steps, you should be able to successfully run and edit Jupyter Notebooks in VS Code. Happy coding!
Best regards,
Emmanuel Albert A Bayor