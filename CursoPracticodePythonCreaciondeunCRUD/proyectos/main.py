import sys

clients = [
        {
            "name":"pablo",
            "company":"Google",
            "email":"pablo@google.com",
            "position":"Software Engineer",
        },
        {
            "name":"Ricardo",
            "company":"Facebook",
            "email":"recardo@facebook.com",
            "position":"Data engineer",
        }
        ]

def create_client(client):
    global clients

    if client not in clients:
        clients.append(client)

    else:
        print("Client already is in client's list")


def search_client(client_name):

    for client in clients:
        if client != client_name:
            continue
        else:
            return True


def list_clients():
    for idx, client in enumerate(clients):
        print("{uid} | {name} | {company} | {email} | {position}".format(
            uid = idx,
            name = client["name"],
            company = client["company"],
            email = client["email"],
            position = client["position"]))


def update_client(client_name, update_client_name):
    global clients

    if len(clients) - 1 >= clients:
        clients[index] = updated_name

    else:
        print("Client is not in client list")


def delete_client(client_name):
    global clients

    if idx, client in enumerate(clients):
        if idx == client_id:
            del clients[idx]
            break


def _get_client_field(field_name):
    field = None

    while not field:
        field = input("What is the clinet {}? ".format(field_name))

    return field

def _get_client_name():
    client_name = None

    while not client_name:
        client_name = input("What is the client name? ")
        
        if client_name == "exit":
            client_name = None
            break

    if not client_name:
        sys.exit()

    return client_name

def _print_welcome():
    print("WELCOME TO PLATZI VENTAS")
    print("*" * 50)
    print("What would you like to do today?")
    print("[C]reate client")
    print("[L]ist client")
    print("[U]pdate client")
    print("[D]elete client")
    print("[S]earch client")


if __name__ == '__main__':
    _print_welcome()

    command = input()
    command = command.upper()

    if command == "C":
        client = {
                "name" : _get_client_field("name"),
                "company" : _get_client_field("company"),
                "email" : _get_client_field("email"),
                "position" : _get_client_field("position"),
        }
        create_client(client)
        list_clients()

    elif command == "L":
        list_clients()

    elif command == "D":
        client_name = _get_client_name()
        delete_client(client_name)
        print("The client "+ client_name + " a been delete")
        list_clients()

    elif command == "U":
        client_name = _get_client_name()
        update_client_name = input("What is the update client name? ")
        update_client(client_name, update_client_name)
        list_clients()

    elif command == "S":
        client_name = _get_client_name()
        found = search_client(client_name)
        
        if found:
            print("The client is in the client's list")

        else:
            print("The client: {} is not our client's list".format(client_name))

    else:
        print("Invalid command")

    #list_clients()

    #create_client('David')
    
    #list_clients()
