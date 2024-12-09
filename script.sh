#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 [-v] <directory_path>"
    echo "  -v    Enable verbose mode"
    exit 1
}

# Parse options
verbose=0
while getopts ":v" opt; do
    case ${opt} in
        v )
            verbose=1
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if a directory path is provided
if [[ -z "$1" ]]; then
    usage
fi

# Specify the directory containing the text files
directory_path="$1"

# Output CSV and log files
output_file="output.csv"
log_file="script_log.txt"

# Start the log file
{
    echo "Script started at $(date)"
    echo "Directory path: $directory_path"
    echo "Output CSV file: $output_file"
    echo "-------------------------"
} > "$log_file"

# Verbose function
log_verbose() {
    if [[ $verbose -eq 1 ]]; then
        echo "$1"
    fi
    echo "$1" >> "$log_file"
}

# Write the header to the CSV file
echo '"Level1","Level2","File","INDICATION","TECHNIQUE","COMPARISON","FINDINGS","IMPRESSION","GENERIC_REPORT"' > "$output_file"

# Loop through all text files in the specified directory structure
for file in "$directory_path"/p*/p*/s*.txt; do
    # Check if the file exists
    if [[ ! -e "$file" ]]; then
        log_verbose "Warning: File not found - $file"
        continue
    fi

    # Extract path components
    file_path=${file#"$directory_path"/}
    IFS='/' read -r level1 level2 filename <<< "$file_path"
    filename_no_ext=${filename%.txt}

    log_verbose "Processing file: $file"

    # Initialize variables for each header section
    INDICATION=""
    TECHNIQUE=""
    COMPARISON=""
    FINDINGS=""
    IMPRESSION=""
    GENERIC_REPORT=""
    header_found=0

    # Initialize current section tracker
    current_section=""

    # Read the file line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Trim leading and trailing whitespace from the line
        line=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')

        # Detect and set the current section based on header line
        case "$line" in
            INDICATION:*) 
                current_section="INDICATION"
                INDICATION="${line#INDICATION: }"
                header_found=1
                ;;
            TECHNIQUE:*) 
                current_section="TECHNIQUE"
                TECHNIQUE="${line#TECHNIQUE: }"
                header_found=1
                ;;
            COMPARISON:*) 
                current_section="COMPARISON"
                COMPARISON="${line#COMPARISON: }"
                header_found=1
                ;;
            FINDINGS:*) 
                current_section="FINDINGS"
                FINDINGS="${line#FINDINGS: }"
                header_found=1
                ;;
            IMPRESSION:*) 
                current_section="IMPRESSION"
                IMPRESSION="${line#IMPRESSION: }"
                header_found=1
                ;;
            # Append lines to the current section
            *)
                if [[ -n "$current_section" ]]; then
                    eval "$current_section=\"\${$current_section} $line\""
                else
                    GENERIC_REPORT="${GENERIC_REPORT}\n${line}"
                fi
                ;;
        esac
    done < "$file"

    # If no headers were found, process the generic report
    if [[ $header_found -eq 0 ]]; then
        log_verbose "Warning: No standard headers found in file $file"
        
        # Split GENERIC_REPORT by double newlines and take the last section as FINDINGS
        GENERIC_REPORT=$(echo -e "$GENERIC_REPORT" | sed '/^$/d' | awk 'BEGIN{RS="\n\n";ORS="\n\n"} {print}')
        FINDINGS=$(echo -e "$GENERIC_REPORT" | awk 'BEGIN{RS="\n\n";ORS="\n\n"} {last=$0} END{print last}' | sed 's/^[ \t]*//;s/[ \t]*$//')
        GENERIC_REPORT=$(echo -e "$GENERIC_REPORT" | sed '$d' | tr -d '\n' | sed 's/^[ \t]*//;s/[ \t]*$//')
    fi

    # Clean up the extracted content for CSV format
    clean_content() {
        echo "$1" | sed 's/^[ \t]*//;s/[ \t]*$//' | tr -d '\n\r' | sed 's/  */ /g' | sed 's/"/""/g'
    }

    INDICATION=$(clean_content "$INDICATION")
    TECHNIQUE=$(clean_content "$TECHNIQUE")
    COMPARISON=$(clean_content "$COMPARISON")
    FINDINGS=$(clean_content "$FINDINGS")
    IMPRESSION=$(clean_content "$IMPRESSION")
    GENERIC_REPORT=$(clean_content "$GENERIC_REPORT")

    # Escape double quotes in path components
    level1="${level1//\"/\"\"}"
    level2="${level2//\"/\"\"}"
    filename_no_ext="${filename_no_ext//\"/\"\"}"

    # Write the data to the CSV file
    echo "\"$level1\",\"$level2\",\"$filename_no_ext\",\"$INDICATION\",\"$TECHNIQUE\",\"$COMPARISON\",\"$FINDINGS\",\"$IMPRESSION\",\"$GENERIC_REPORT\"" >> "$output_file"
    log_verbose "Finished processing file: $file"

done

# End of script log
log_verbose "Script finished at $(date)"
