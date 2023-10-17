USE metro_cdmx;

ALTER TABLE `stations`
MODIFY `id` BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT,
ADD PRIMERY KEY,

--ADD CONSTRAINT `trains_line_id_foreign`
--FOREIGN KEY (`line_id`) REFERENCES `lines` (`id`);

CONSTRAINT `trains_line_id_foreign` FOREIGN KEY (`line_id`) REFERENCES `lines` (`id`);