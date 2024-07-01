to install req, do while read requirement; do pip install $requirement || echo "Failed to install $requirement"; done < requirements.txt

