def grounds_to_file(lengths, energies, en_lens, durations, J):
  """
  Save three lists as columns in a text file.
  
  Parameters
  ---
    lengths: list
      The lengths of the Heisenberg models
    energies: list
      The GS energies of the Heisenberg models
    en_lens: list
      The energy/lengths of the Heisenberg models
    durations: list
      The durations in seconds of each diagonalization
    J: float
      The magnetic field. Used as filename and saved in header
  """
  filename = f"output/J{J:.5f}.txt"
  
  with open(filename, 'w') as f:
    f.write(f"Heisenberg model lengths, energies, E/L and time to diagonalize (s) in J={J}\n")
    for item1, item2, item3, item4 in zip(lengths, energies, en_lens, durations):
      f.write(f"{item1}\t{item2:.5f}\t{item3:.5f}\t{item4:.5f}\n")