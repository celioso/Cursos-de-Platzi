def check_access(func):
    def wrapper(employee):
        # Comprobar el rol 'admin'
        if employee.get('role') == 'admin':
            return func(employee)
        else:
            print('ACCESO DENEGADO: solo los administradores pueden acceder.')
    return wrapper

@check_access
def dalete_employee(employee):
    print(f'El empleado {employee['name']} ha sido eliminado')

admin = {'name': 'Carlos', 'role':'admin'}
employee = {'name':'Ana', 'role':'employee'}

#dalete_employee(admin)
dalete_employee(employee)
