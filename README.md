Coupling Matrix 

A single-file Python tool for iPad (Pyto) that extracts the coupling matrix of an N-pole inline bandpass filter from a measured Touchstone (.s2p) file. It needs only NumPy, SciPy and Matplotlib.

Features

	•	Strict Touchstone 1.x reader. It rejects unsupported or malformed files with clear errors and keeps the file's reference impedance.
	•	Robust extraction. A global magnitude search is followed by complex S11/S21/S22 refinement. It fits reference-plane delays and flags ambiguous or poorly determined fits.
	•	Transmission zeros. Optional signed cross-couplings (for example 2-4, or 1-3 and 3-5) are supported at any order.
	•	Synthesise target. Given a return loss and a zero value or frequency, it solves a matched target design and saves it as a target file.
	•	Tuning advisor. It compares the extracted matrix with a target and ranks parameters by error against tolerance. It suggests one electrical move at a time and exports .txt and .csv reports.
	•	Live editor. An embedded WebView page lets you edit the matrix and watch S11 and S21 update against the measurement and target. It has Next best move, Auto-tune and a Save tuned.s2p button.
