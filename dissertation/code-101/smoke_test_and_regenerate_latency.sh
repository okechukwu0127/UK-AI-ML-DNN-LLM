#!/usr/bin/env bash
set -euo pipefail

echo "What the smoke script does:"
echo " "
echo "sends a random mix of benign and malicious requests to the normal endpoints"
echo "waits a random number of seconds between requests"
echo "hits dataset_batch_detect once per run"
echo "regenerates the latency report afterward"

BASE_URL="${BASE_URL:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/sqli/bin/python}"
PLOT_ENV="${PLOT_ENV:-/private/tmp}"
RANDOM_REQUESTS="${RANDOM_REQUESTS:-20}"
SLEEP_MIN="${SLEEP_MIN:-1}"
SLEEP_MAX="${SLEEP_MAX:-5}"
DATASET_BATCH_SAMPLE_SIZE="${DATASET_BATCH_SAMPLE_SIZE:-150}"

check_server() {
  local ports=("5000" "5100")
  if [[ -n "$BASE_URL" ]]; then
    if curl -fsS "$BASE_URL/health" >/dev/null; then
      return 0
    fi
    echo "Server is not reachable at $BASE_URL."
    exit 1
  fi

  for port in "${ports[@]}"; do
    if curl -fsS "http://localhost:${port}/health" >/dev/null; then
      BASE_URL="http://localhost:${port}"
      echo "Using $BASE_URL"
      return 0
    fi
  done

  echo "Server is not reachable on localhost:5000 or localhost:5100."
  echo "Start one of the Flask runners first:"
  echo "  cd $PROJECT_DIR"
  echo "  MPLCONFIGDIR=/private/tmp ./sqli/bin/python -c \"from sql_injection_middleware import app; app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)\""
  echo "or:"
  echo "  python app.py"
  exit 1
}

hit_endpoint() {
  local label="$1"
  local method="$2"
  local url="$3"
  local data="${4:-}"

  echo "Hitting: $label"
  if [[ "$method" == "GET" ]]; then
    curl -sS -o /dev/null -w "HTTP %{http_code}\n" "$url"
  else
    curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X "$method" "$url" -H 'Content-Type: application/json' -d "$data"
  fi
}

random_sleep() {
  local min="$1"
  local max="$2"
  if (( max <= min )); then
    sleep "$min"
    return
  fi

  local span=$((max - min + 1))
  local delay=$((RANDOM % span + min))
  sleep "$delay"
}

run_random_traffic() {
  local benign_gets=(
    "$BASE_URL/api/users?user_id=123&name=john"
    "$BASE_URL/api/search?q=laptop&category=electronics"
    "$BASE_URL/api/products/1"
  )
  local benign_posts=(
    '{"username":"admin","password":"secret123"}'
    '{"order_id":123,"product":"keyboard"}'
  )
  local malicious_gets=(
    "$BASE_URL/api/users?user_id=1%20OR%201=1--"
    "$BASE_URL/api/search?q=1%20UNION%20SELECT%20username,password%20FROM%20users"
    "$BASE_URL/api/products/1%27%20OR%20%271%27=%271"
  )
  local malicious_posts=(
    '{"username":"admin'\''--","password":"anything"}'
    '{"order_id":"1; DROP TABLE orders;--"}'
    '{"sql_query":"SELECT * FROM users WHERE id = 1 OR 1=1 --"}'
  )

  echo "Running randomized request traffic: $RANDOM_REQUESTS requests"
  for ((i=1; i<=RANDOM_REQUESTS; i++)); do
    local pick=$((RANDOM % 2))
    local route_pick=$((RANDOM % 5))
    local method_pick=$((RANDOM % 2))

    if (( pick == 0 )); then
      if (( method_pick == 0 )); then
        local url="${benign_gets[$((RANDOM % ${#benign_gets[@]}))]}"
        hit_endpoint "benign-random-$i" GET "$url"
      else
        local body="${benign_posts[$((RANDOM % ${#benign_posts[@]}))]}"
        case "$route_pick" in
          0) hit_endpoint "benign-random-$i" POST "$BASE_URL/api/login" "$body" ;;
          1) hit_endpoint "benign-random-$i" POST "$BASE_URL/api/orders" "$body" ;;
          2) hit_endpoint "benign-random-$i" POST "$BASE_URL/detect_single" '{"sql_query":"SELECT id FROM users WHERE username = \"john\""}' ;;
          3) hit_endpoint "benign-random-$i" POST "$BASE_URL/batch_detect" '{"queries":["SELECT * FROM users","SELECT * FROM products"]}' ;;
          *) hit_endpoint "benign-random-$i" GET "$BASE_URL/api/users?user_id=123&name=john" ;;
        esac
      fi
    else
      if (( method_pick == 0 )); then
        local url="${malicious_gets[$((RANDOM % ${#malicious_gets[@]}))]}"
        hit_endpoint "malicious-random-$i" GET "$url"
      else
        local body="${malicious_posts[$((RANDOM % ${#malicious_posts[@]}))]}"
        case "$route_pick" in
          0) hit_endpoint "malicious-random-$i" POST "$BASE_URL/api/login" "$body" ;;
          1) hit_endpoint "malicious-random-$i" POST "$BASE_URL/api/orders" "$body" ;;
          2) hit_endpoint "malicious-random-$i" POST "$BASE_URL/detect_single" '{"sql_query":"1 UNION SELECT username,password FROM users"}' ;;
          3) hit_endpoint "malicious-random-$i" POST "$BASE_URL/batch_detect" '{"queries":["1 UNION SELECT username,password FROM users","DROP TABLE users;--"]}' ;;
          *) hit_endpoint "malicious-random-$i" GET "$BASE_URL/api/products/1%27%20OR%20%271%27=%271" ;;
        esac
      fi
    fi

    random_sleep "$SLEEP_MIN" "$SLEEP_MAX"
  done
}

check_server

hit_endpoint "health" GET "$BASE_URL/health"
run_random_traffic
hit_endpoint "dataset batch" GET "$BASE_URL/dataset_batch_detect?sample_size=$DATASET_BATCH_SAMPLE_SIZE"
hit_endpoint "admin stats" GET "$BASE_URL/admin/stats"
hit_endpoint "admin logs" GET "$BASE_URL/admin/logs?limit=10"


echo "Regenerating latency and dissertation reports..."
MPLCONFIGDIR="$PLOT_ENV" "$PYTHON_BIN" "$PROJECT_DIR/generate_methodology_figures.py"

echo "Done."
echo "Latency report:"
echo "  $PROJECT_DIR/sql_injection_plots/inference_latency_summary.csv"
echo "  $PROJECT_DIR/sql_injection_plots/inference_latency_summary.png"
