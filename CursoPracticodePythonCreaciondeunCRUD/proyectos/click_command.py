import click
@click.command()
#basic options
@click.option("--name", "-n", default="John", help="Firstname description") #help nos ayuda a ver una descripcion de este comando -n --name se puede cambiar el nombre de la variable name

#Multiple Values
@click.option("--salary", "-s",nargs=2, help= "Your Monthly Salary", type=int) # para que se pueda colocar dos variables en nombre EJ: py <nom_archivo> -n carlos luis 

# Multiple Options
@click.option("--location", "-l", help ="Placce You ve Visited", multiple=True)

def main(name, salary,location):
    click.echo("Hello World My Name is {}, My salary is {}.".format(name, sum(salary)))
    click.echo("\n".join(location))


if __name__ == "__main__":
    main()