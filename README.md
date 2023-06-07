# ECS171-Proj
To run Pokemon interface code,
1. Download the repo.
2. pip install flask (Prevents --> ModuleNotFoundError: No module named 'flask')
3. pip install beautifulsoup4 (Prevents --> ModuleNotFoundError: No module named 'bs4')
4. Install other packages if error stating ModuleNotFoundError
5. cd Interface 
6. Run "python main.py"
7. When see this statement, "Running on http://127.0.0.1:5000", copy the link http://127.0.0.1:5000 and paste URL in address bar (or Ctrl + Click).
8. Once entered, interface should work.


VIDEO SHOWCASING CODE AND WEB INTERFACE HERE: https://vimeo.com/833874617/fc7587dd8f?share=copy
The only changes in code was:
1. deletion of file tempCodeRunnerFile.py because it was used as a temporary tester. Interface code works is not related or affected by it. 
2. debug=false in main.py for Windows but debug=true should work for Mac. Depends on user's computer.

WARNING: Don't repeatedbly spam click on new types without letting a previous one finish, it takes ~10 seconds for the trained models to be loaded and accessed for every type you select.
