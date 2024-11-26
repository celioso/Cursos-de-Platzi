SELECT 
    c.customerid AS cod_cliente, 
    p.firstname AS nombre, 
    p.lastname AS apellido,
    p.firstname || ' ' || p.lastname AS nombre_completo,
    CASE 
        WHEN pp.phonenumbertypeid = 1 THEN pp.phonenumber 
        ELSE NULL 
    END AS numero_telefono_celular,
    CASE 
        WHEN pp.phonenumbertypeid = 2 THEN pp.phonenumber 
        ELSE NULL 
    END AS numero_telefono_casa,
    CASE 
        WHEN pp.phonenumbertypeid = 3 THEN pp.phonenumber 
        ELSE NULL 
    END AS numero_telefono_trabajo,
    a.city
FROM sales.customer c
LEFT JOIN person.person p 
    ON c.personid = p.businessentityid
LEFT JOIN person.personphone pp 
    ON p.businessentityid = pp.businessentityid
LEFT JOIN person.businessentity b
    ON p.businessentityid = b.businessentityid
LEFT JOIN person.businessentityaddress b2
    ON b.businessentityid = b2.businessentityid 
    AND b2.addresstypeid = 2
LEFT JOIN person.address a
    ON b2.addressid = a.addressid;

