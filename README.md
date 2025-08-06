# parallel-chess-programming-engine

Here are the libraries you're using in your code and their purposes:

tkinter: For creating the graphical user interface (GUI), including the main window, canvas, labels, and dialog boxes.

tkinter.simpledialog: A submodule of tkinter specifically for creating simple pop-up dialog windows to get user input (like choosing a game mode).

chess: The Python Chess library (imported as chess_game). This is the core library for all chess-related logic, such as representing the board, handling moves, and checking for game-ending conditions.


random: Used to add randomness, specifically for shuffling legal moves in your minimax algorithm.

torch: The PyTorch library, a powerful deep learning framework. You're using it to perform parallel computations on the CPU and GPU to benchmark the board evaluation function.

threading: Allows you to run the CPU and GPU benchmarks concurrently, so the program can perform both tasks at the same time without one blocking the other.

time: Used for timing the performance of your evaluation functions, which is essential for the benchmark.

logging: For writing detailed messages and game history to a log file, helping you track what happens during a game.

datetime: A standard library used to create a unique timestamp for your log file name.

pygame: A library for game development. You're using its mixer module to handle and play sound effects for moves and captures.

platform: A standard library that provides access to the platform's underlying data, like the processor name, which you use to get the CPU information for your benchmark output.

   
   for installing the Python libraries :-
   
   pip install python-chess torch pygame
