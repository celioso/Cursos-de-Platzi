--Eliminar tablas
--DROP TABLE dim_clientes CASCADE;
--DROP TABLE dim_productos CASCADE;
--DROP TABLE dim_territorios CASCADE;
--DROP TABLE dim_vendedores CASCADE;
--DROP TABLE fact_ventas CASCADE;

-- Tabla dim_clientes
CREATE TABLE dim_clientes (
    id_cliente INTEGER,
    codigo_cliente VARCHAR(10),
    nombre VARCHAR(50),
    apellido VARCHAR(50),
    nombre_completo VARCHAR(100),
    numero_telefono_celular VARCHAR(20),
    numero_telefono_casa VARCHAR(20),
    numero_telefono_trabajo VARCHAR(20),
    ciudad_casa VARCHAR(50),
    fecha_carga TIMESTAMP,
    fecha_actualizacion TIMESTAMP,
    PRIMARY KEY (id_cliente)
);

-- Tabla dim_productos
CREATE TABLE dim_productos (
    id_producto INTEGER PRIMARY KEY,
    codigo_producto VARCHAR(20),
    nombre VARCHAR(50),
    color VARCHAR(50),
    tamaño VARCHAR(50),
    categoria VARCHAR(50),
    fecha_carga TIMESTAMP,
    fecha_actualizacion TIMESTAMP
);

-- Tabla dim_vendedores
CREATE TABLE dim_vendedores (
    id_vendedor INTEGER PRIMARY KEY,
    codigo_vendedor VARCHAR(20),
    identificacion VARCHAR(10),
    nombre VARCHAR(50),
    apellido VARCHAR(50),
    nombre_completo VARCHAR(100),
    rol VARCHAR(10),
    fecha_nacimiento DATE,
    genero VARCHAR(10),
    ind_activo BOOLEAN,
    fecha_inicio DATE,
    fecha_fin DATE,
    ind_bono BOOLEAN,
    fecha_carga TIMESTAMP
);

-- Tabla dim_territorios
CREATE TABLE dim_territorios (
    id_territorio INTEGER PRIMARY KEY,
    codigo_territorio VARCHAR(10),
    nombre VARCHAR(50),
    continente VARCHAR(10),
    fecha_carga TIMESTAMP,
    fecha_actualizacion TIMESTAMP
);

-- Tabla fact_ventas
CREATE TABLE fact_ventas (
    id_venta INTEGER PRIMARY KEY NOT NULL,
    codigo_venta_detalle VARCHAR(50) NOT NULL,
    codigo_venta_encabezado VARCHAR(50) NOT NULL,
    id_fecha INTEGER,
    id_territorio INTEGER,
    id_cliente INTEGER,
    id_vendedor INTEGER,
    id_producto INTEGER,
    cantidad INTEGER,
    valor DECIMAL(10, 2),
    descuento DECIMAL(10, 2),
    fecha_carga TIMESTAMP,
    fecha_actualizacion TIMESTAMP
);

-- Tabla dim_tiempo
CREATE TABLE dim_tiempo (
    id_fecha INTEGER NOT NULL,
    fecha DATE NOT NULL,
    dia SMALLINT NOT NULL,
    mes SMALLINT NOT NULL,
    ano SMALLINT NOT NULL,
    dia_semana SMALLINT NOT NULL,
    dia_ano SMALLINT NOT NULL,
    PRIMARY KEY (id_fecha)
);

-- Relaciones (llaves foráneas)
ALTER TABLE fact_ventas
    ADD CONSTRAINT territorios_ventas_fk FOREIGN KEY (id_territorio)
    REFERENCES dim_territorios (id_territorio);

ALTER TABLE fact_ventas
    ADD CONSTRAINT clientes_ventas_fk FOREIGN KEY (id_cliente)
    REFERENCES dim_clientes (id_cliente);

ALTER TABLE fact_ventas
    ADD CONSTRAINT vendedores_ventas_fk FOREIGN KEY (id_vendedor)
    REFERENCES dim_vendedores (id_vendedor);

ALTER TABLE fact_ventas
    ADD CONSTRAINT productos_ventas_fk FOREIGN KEY (id_producto)
    REFERENCES dim_productos (id_producto);
