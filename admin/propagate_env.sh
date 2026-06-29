echo "Searching for pyproject.toml files..."
pyproject_files=$(find . -name "pyproject.toml" -type f)
echo "Found files:"
echo "$pyproject_files"
echo ""

for pyproject in $pyproject_files; do

    root=$(dirname "$pyproject" | sed "s|\./||g")
    
    cp -v admin/.env ${root} 
    
done