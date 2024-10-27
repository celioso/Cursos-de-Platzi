class Employee:
    def __init__(self, name, last_name,*args, **kwargs):
        self.name = name
        self.last_name=last_name
        self.skills=args
        self.detail=kwargs

    def show_employee(self):
        print(f'Employee: {self.name} {self.last_name}')
        print(f'Skills: {self.skills}')
        print(f'Details: {self.detail}')

employee = Employee('Carlos', 'Castro', 'Pytho', 'Java','C++', age=30, city='bogota')
employee.show_employee()