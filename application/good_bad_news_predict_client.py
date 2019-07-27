#! usr/bin/env python
# -*- coding : utf-8 -*-


import requests
import json


def main():
    url = "http://localhost:4000/jsonrpc"
    headers = {'content-type': 'application/json'}

    # Example echo method
    payload = {
        "method": "get_request",
        "params": [' @Amy_Siskind @drspl5 I m sure he will welcome Hurricane Irma on the horizon as something to distract. '],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url, data=json.dumps(payload), headers=headers).json()

    print(response)

if __name__ == "__main__":
    main()