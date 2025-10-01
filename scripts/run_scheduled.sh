#!/bin/bash
# Scheduled trading script - runs daily at 9:50 AM ET
# This script is called by cron

set -e  # Exit on error

# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs/live"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/trading_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Log start
echo "========================================" | tee -a "${LOG_FILE}"
echo "Starting trading run at $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Change to project directory
cd "${PROJECT_DIR}"

# Ensure Docker container is running
docker-compose up -d >> "${LOG_FILE}" 2>&1

# Run trading script inside Docker container
if docker-compose exec -T trading-bot python scripts/run_live.py --mode paper >> "${LOG_FILE}" 2>&1; then
    echo "SUCCESS: Trading run completed at $(date)" | tee -a "${LOG_FILE}"
    EXIT_CODE=0
else
    echo "ERROR: Trading run failed at $(date)" | tee -a "${LOG_FILE}"
    EXIT_CODE=1
fi

# Cleanup old logs (keep last 30 days)
find "${LOG_DIR}" -name "trading_*.log" -mtime +30 -delete

# Send notification on error (optional - configure below)
if [ $EXIT_CODE -ne 0 ]; then
    # Uncomment and configure for email notifications:
    # echo "Trading run failed. See ${LOG_FILE}" | mail -s "Trading Error" your-email@example.com

    # Or use a webhook (Slack, Discord, etc):
    # curl -X POST -H 'Content-type: application/json' \
    #   --data "{\"text\":\"Trading run failed at $(date)\"}" \
    #   YOUR_WEBHOOK_URL

    echo "ERROR: Check ${LOG_FILE} for details"
fi

exit $EXIT_CODE
