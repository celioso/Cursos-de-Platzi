from listener import Listener


class AccountabilityListener[T](Listener):
    def notify(self, event: T):
        print("Notificando el evento {event}")
