
"""def dynamodb_events(event, context):
    print('Nuevo Evento')
    print(event)"""



def dynamodb_events(event, context):
    try:
        for record in event['Records']:
            event_name = record.get('eventName')
            
            if event_name == 'INSERT':
                manejar_insert(record)
            elif event_name == 'MODIFY':
                manejar_update(record)
            elif event_name == 'REMOVE':
                manejar_delete(record)
                
    except Exception as e:
        print(f"Error procesando evento: {e}")
        return 'Error'
    
    return 'Listo'

def manejar_insert(record):
    print('🟢 Llegó un evento INSERT')
    print(record)

def manejar_update(record):
    print('🟡 Llegó un evento MODIFY')
    print(record)

def manejar_delete(record):
    print('🔴 Llegó un evento REMOVE')
    print(record)
