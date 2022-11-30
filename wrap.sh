
#!/bin/bash

# Start the first process
python backend.py &
  
# Start the second process
streamlit run frontend.py &
  
wait
