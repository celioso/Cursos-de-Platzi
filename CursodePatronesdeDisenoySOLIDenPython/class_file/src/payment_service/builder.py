from dataclasses import dataclass
from typing import Optional, Self

from service import PaymentService
from commons import PaymentData, CustomerData
from loggers import TransactionLogger
from notifiers import NotifierProtocol, EmailNotifier, SMSNotifier
from factory import PaymentProcessorFactory
from processors import (
    PaymentProcessorProtocol,
    RecurringPaymentProcessorProtocol,
    RefundProcessorProtocol,
)

from listeners import ListenerManager, AccountabilityListener
from validators import CustomerValidator, PaymentDataValidator, CustomerHandler, ChainHandler

@dataclass
class PaymentServiceBuilder:
    payment_processor: Optional[PaymentProcessorProtocol] = None
    notifier: Optional[NotifierProtocol] = None
    #customer_validator: Optional[CustomerValidator] = None
    validator: Optional[ChainHandler] = None
    #payment_validator: Optional[PaymentDataValidator] = None
    logger: Optional[TransactionLogger] = None
    listener: Optional[ListenerManager] = None
    refund_processor: Optional[RefundProcessorProtocol] = None
    recurring_processor: Optional[RecurringPaymentProcessorProtocol] = None

    def set_logger(self) ->Self:
        self.logger = TransactionLogger()
        return self
    
    def set_payment_validator(self)-> Self:
        self.payment_validator = PaymentDataValidator()
        return self
    
#    def set_customer_validator(self)-> Self:
#        self.customer_validator = CustomerValidator()
#        return self
    
#    def set_payment_processor(self, payment_data: PaymentData) -> Self:
#        self.payment_processor = (
#            PaymentProcessorFactory.create_payment_processor(payment_data)
#        )
#        return self
    def set_chain_of_validations(self) -> Self:
        customer_handler = CustomerHandler()
        customer_handler_2 = CustomerHandler()
        customer_handler.set_next(customer_handler_2)

        self.validator = customer_handler_2

        return self

    def set_notifier(self, customer_data: CustomerData) -> Self:
        if customer_data.contact_info.email:
            self.notifier = EmailNotifier()
            return self

        if customer_data.contact_info.phone:
            self.notifier = SMSNotifier(gateway="MyCustomGateway")
            return self

        raise ValueError("No se puede seleccionar clase de notificación")
    
    def set_listeners(self):
        listener = ListenerManager()
        accontability_listener = AccountabilityListener()
        listener.subscribe(accontability_listener)

        self.listener = listener

    def build(self):
        if not all(
            [
            self.payment_processor,
            self.notifier,
            self.validator,
            self.logger, 
            self.listener
            ]
        ): 
            misssing = [
                name
                for name, value in [
                    ("payment_processor", self.payment_processor),
                    ("notifier", self.notifier),
                    ("validator", self.validator),
                    ("logger", self.logger),
                    ("listener", self.listener)
                ]
                if value is None
            ]
            raise ValueError(f"Missing dependencies: {misssing}")
        
        return PaymentService(
            payment_processor=self.payment_processor, # type: ignore
            validators=self.validator,
            notifier=self.notifier,
            logger=self.logger,
            listeners=self.listener
        )