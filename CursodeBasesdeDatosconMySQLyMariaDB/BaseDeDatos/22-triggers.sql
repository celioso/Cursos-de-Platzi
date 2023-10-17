USE metro_cdmx;

CREATE TRIGGER IF NOT EXISTS Supdate_updated_at_field
BEFORE UPDATE 
ON `lines`
FOR EACH ROW 
SET NEW.updated_at=NOW()

CREATE TRIGGER IF NOT EXISTS update_updated_at_field
BEFORE UPDATE 
ON `lines_stations`
FOR EACH ROW 
SET NEW.updated_at=NOW();

CREATE TRIGGER IF NOT EXISTS update_updated_at_field
BEFORE UPDATE 
ON `lacations`
FOR EACH ROW 
SET NEW.updated_at=NOW();

CREATE TRIGGER IF NOT EXISTS update_updated_at_field
BEFORE UPDATE 
ON `stations`
FOR EACH ROW 
SET NEW.updated_at=NOW();

CREATE TRIGGER IF NOT EXISTS update_updated_at_field
BEFORE UPDATE 
ON `trains`
FOR EACH ROW 
SET NEW.updated_at=NOW();


-- otra forma de que el trigger no se bloque 

DROP TRIGGER IF EXISTS update_updated_at_field
CREATE TRIGGER update_updated_at_field
BEFORE UPDATE ON `lines`
FOR EACH ROW 
SET NEW.updated_at = NOW();


--para no utilizar triggers 
ALTER TABLE `stations`
MODIFY COLUMN `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;

ALTER TABLE `lines`
MODIFY COLUMN `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;

ALTER TABLE `trains`
MODIFY COLUMN `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;