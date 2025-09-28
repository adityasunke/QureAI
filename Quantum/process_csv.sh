#!/bin/bash

# Set the relative path to the CSV files
CSV_DIR="Haemophilus_influenzae_3D_Coords"
OUTPUT_DIR="Outputs"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the CSV directory exists
if [ ! -d "$CSV_DIR" ]; then
    echo "Error: Directory '$CSV_DIR' does not exist."
    exit 1
fi

# Check if there are any CSV files in the directory
if ! ls "$CSV_DIR"/*.csv 1> /dev/null 2>&1; then
    echo "Error: No CSV files found in '$CSV_DIR'."
    exit 1
fi

# Counter for processed files
count=0

echo "Starting to process CSV files from: $CSV_DIR"
echo "Output will be stored in: $OUTPUT_DIR"
echo "----------------------------------------"

# Loop through all CSV files in the directory
for csv_file in "$CSV_DIR"/*.csv; do
    # Get the filename without the path
    filename=$(basename "$csv_file")
    echo "Processing: $filename"
    
    # Run the Python script with the CSV file as parameter
    # Store any console output for error checking
    log_file="$OUTPUT_DIR/${filename%.csv}_log.txt"
    
    if python vqe.py "$csv_file" > "$log_file" 2>&1; then
        echo "  ✓ Python script executed successfully"
        
        # Move generated CSV and PNG files to Outputs directory
        # Look for recently created files (within the last minute)
        moved_files=0
        
        # Find and move CSV files (excluding the original input file)
        for output_csv in *.csv; do
            if [ "$output_csv" != "*.csv" ] && [ "$output_csv" != "$csv_file" ] && [ "$output_csv" -nt "$csv_file" ]; then
                mv "$output_csv" "$OUTPUT_DIR/"
                echo "    → Moved CSV: $output_csv"
                ((moved_files++))
            fi
        done
        
        # Find and move PNG files
        for output_png in *.png; do
            if [ "$output_png" != "*.png" ] && [ "$output_png" -nt "$csv_file" ]; then
                mv "$output_png" "$OUTPUT_DIR/"
                echo "    → Moved PNG: $output_png"
                ((moved_files++))
            fi
        done
        
        if [ $moved_files -gt 0 ]; then
            echo "  ✓ Success - $moved_files output file(s) moved to $OUTPUT_DIR"
            ((count++))
        else
            echo "  ⚠ Warning - No output files found to move"
        fi
        
        # Clean up log file if it's empty or only contains normal output
        if [ ! -s "$log_file" ]; then
            rm "$log_file"
        fi
    else
        echo "  ✗ Error processing $filename - Check $log_file for details"
    fi
    
    echo ""
done

echo "----------------------------------------"
echo "Processing complete. $count files processed successfully."