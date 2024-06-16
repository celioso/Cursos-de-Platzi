import requests

def login():
    login_url = "http://mini-challenge.foris.ai/login"
    login_data = {
        "username": "foris_challenge",
        "password": "ForisChallenge"
    }

    try:
        response = requests.post(login_url, json=login_data)
        response.raise_for_status()
        return response.json().get("auth_token")
    except requests.exceptions.RequestException as e:
        print(f"Login request failed: {e}")
        return None
    except ValueError:
        print("Login response is not valid JSON.")
        return None


def get_challenge(auth_token):
    challenge_url = "http://mini-challenge.foris.ai/desafio"
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        response = requests.get(challenge_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Challenge request failed: {e}")
        return None
    except ValueError:
        print("Challenge response is not valid JSON.")
        return None


def get_dumps(auth_token, dump_type):
    dumps_url = f"http://mini-challenge.foris.ai/volcados/{dump_type}"
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        response = requests.get(dumps_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Dumps request failed: {e}")
        return None
    except ValueError:
        print("Dumps response is not valid JSON.")
        return None


def validate_answer(auth_token, number_of_groups, answer):
    validate_url = "http://mini-challenge.foris.ai/validar"
    validate_data = {
        "number_of_groups": number_of_groups,
        "answer": answer
    }
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        response = requests.post(validate_url, json=validate_data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Validation request failed: {e}")
        return None
    except ValueError:
        print("Validation response is not valid JSON.")
        return None


def main():
    auth_token = login()
    if not auth_token:
        return

    challenge_data = get_challenge(auth_token)
    if not challenge_data:
        return

    print("Challenge data:", challenge_data)


    dump_type = challenge_data.get('dump_type')  
    if not dump_type:
        print("No dump type found in challenge data.")
        return

    dumps_data = get_dumps(auth_token, dump_type)
    if not dumps_data:
        return

    print("Dumps data:", dumps_data)

    number_of_groups = 26  
    answer = "your_answer"  

    validation_response = validate_answer(auth_token, number_of_groups, answer)
    if validation_response:
        print("Validation successful:", validation_response)
    else:
        print("Validation failed.")

if __name__ == "__main__":
    main()