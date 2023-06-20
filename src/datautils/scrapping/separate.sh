shopt -s nullglob

# Create or clear the train.txt and test.txt files
> train.txt
> test.txt

# Find all JSON files recursively under the current directory and save the relative paths
find . -type f -name "*.json" | while read -r file; do
    # Generate a random number between 1 and 10
    random=$((RANDOM % 10 + 1))
    
    # Determine the destination file based on the random number
    if [[ $random -le 9 ]]; then
        dest="train.txt"
    else
        dest="test.txt"
    fi
    
    # Get the relative file path
    rel_path="${file#./}"
    
    # Append the relative file path to the corresponding destination file
    echo "$rel_path" >> "$dest"
done
