#!/bin/bash
source ./bin/activate
hap run -n odour-service -- uvicorn server:app --host 0.0.0.0 --port 6801
