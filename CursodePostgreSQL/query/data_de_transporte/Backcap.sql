PGDMP      (                 }         
   transporte    17.2    17.2 7    �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                           false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                           false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                           false            �           1262    16388 
   transporte    DATABASE     }   CREATE DATABASE transporte WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'Spanish_Spain.1252';
    DROP DATABASE transporte;
                     postgres    false                        3079    16773    dblink 	   EXTENSION     :   CREATE EXTENSION IF NOT EXISTS dblink WITH SCHEMA public;
    DROP EXTENSION dblink;
                        false            �           0    0    EXTENSION dblink    COMMENT     _   COMMENT ON EXTENSION dblink IS 'connect to other PostgreSQL databases from within a database';
                             false    2                        3079    16819    fuzzystrmatch 	   EXTENSION     A   CREATE EXTENSION IF NOT EXISTS fuzzystrmatch WITH SCHEMA public;
    DROP EXTENSION fuzzystrmatch;
                        false            �           0    0    EXTENSION fuzzystrmatch    COMMENT     ]   COMMENT ON EXTENSION fuzzystrmatch IS 'determine similarities and distance between strings';
                             false    3            �            1255    16767    importantpl()    FUNCTION     L  CREATE FUNCTION public.importantpl() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    rec RECORD;
    contador integer := 0;
BEGIN
    FOR rec IN SELECT * FROM pasajeros LOOP
        contador := contador + 1;
    END LOOP;
    INSERT INTO conteo_pasajeros (total, tiempo)
	VALUES (contador, now());
	RETURN NEW;
END;

$$;
 $   DROP FUNCTION public.importantpl();
       public               postgres    false            �            1259    16757    conteo_pasajeros    TABLE     u   CREATE TABLE public.conteo_pasajeros (
    id integer NOT NULL,
    total integer,
    tiempo time with time zone
);
 $   DROP TABLE public.conteo_pasajeros;
       public         heap r       postgres    false            �            1259    16756    conteo_pasajeros_id_seq    SEQUENCE     �   CREATE SEQUENCE public.conteo_pasajeros_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 .   DROP SEQUENCE public.conteo_pasajeros_id_seq;
       public               postgres    false    231            �           0    0    conteo_pasajeros_id_seq    SEQUENCE OWNED BY     S   ALTER SEQUENCE public.conteo_pasajeros_id_seq OWNED BY public.conteo_pasajeros.id;
          public               postgres    false    230            �            1259    16624    viajes    TABLE     �   CREATE TABLE public.viajes (
    id_viaje integer NOT NULL,
    n_documento integer NOT NULL,
    id_trayecto integer NOT NULL,
    inicio time with time zone NOT NULL,
    fin time with time zone NOT NULL
);
    DROP TABLE public.viajes;
       public         heap r       postgres    false            �            1259    16708    despues_noche_mview    MATERIALIZED VIEW     �   CREATE MATERIALIZED VIEW public.despues_noche_mview AS
 SELECT id_viaje,
    n_documento,
    id_trayecto,
    inicio,
    fin
   FROM public.viajes
  WHERE (inicio > '22:00:00-05'::time with time zone)
  WITH NO DATA;
 3   DROP MATERIALIZED VIEW public.despues_noche_mview;
       public         heap m       postgres    false    225    225    225    225    225            �            1259    16615 
   estaciones    TABLE     �   CREATE TABLE public.estaciones (
    id_estacion integer NOT NULL,
    nombre character varying NOT NULL,
    direccion character(255) NOT NULL
);
    DROP TABLE public.estaciones;
       public         heap r       postgres    false            �            1259    16614    estaciones_id_estacion_seq    SEQUENCE     �   CREATE SEQUENCE public.estaciones_id_estacion_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 1   DROP SEQUENCE public.estaciones_id_estacion_seq;
       public               postgres    false    223            �           0    0    estaciones_id_estacion_seq    SEQUENCE OWNED BY     Y   ALTER SEQUENCE public.estaciones_id_estacion_seq OWNED BY public.estaciones.id_estacion;
          public               postgres    false    222            �            1259    16597 	   pasajeros    TABLE     �   CREATE TABLE public.pasajeros (
    n_documento integer NOT NULL,
    nombre character varying,
    direccion_residencia character varying,
    fecha_nacimiento date
);
    DROP TABLE public.pasajeros;
       public         heap r       postgres    false            �            1259    16704 
   rango_view    VIEW       CREATE VIEW public.rango_view AS
 SELECT n_documento,
    nombre,
    direccion_residencia,
    fecha_nacimiento,
        CASE
            WHEN (fecha_nacimiento > '2012-01-01'::date) THEN 'Niño'::text
            ELSE 'Mayor'::text
        END AS "case"
   FROM public.pasajeros;
    DROP VIEW public.rango_view;
       public       v       postgres    false    219    219    219    219            �            1259    16631 	   trayectos    TABLE     �   CREATE TABLE public.trayectos (
    id_trayecto integer NOT NULL,
    id_estacion integer NOT NULL,
    id_tren integer NOT NULL,
    nombre character varying NOT NULL
);
    DROP TABLE public.trayectos;
       public         heap r       postgres    false            �            1259    16630    trayectos_id_trayecto_seq    SEQUENCE     �   CREATE SEQUENCE public.trayectos_id_trayecto_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 0   DROP SEQUENCE public.trayectos_id_trayecto_seq;
       public               postgres    false    227            �           0    0    trayectos_id_trayecto_seq    SEQUENCE OWNED BY     W   ALTER SEQUENCE public.trayectos_id_trayecto_seq OWNED BY public.trayectos.id_trayecto;
          public               postgres    false    226            �            1259    16606    trenes    TABLE     r   CREATE TABLE public.trenes (
    id_tren integer NOT NULL,
    modelo character varying,
    capacidad integer
);
    DROP TABLE public.trenes;
       public         heap r       postgres    false            �            1259    16605    trenes_id_trenes_seq    SEQUENCE     �   CREATE SEQUENCE public.trenes_id_trenes_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.trenes_id_trenes_seq;
       public               postgres    false    221            �           0    0    trenes_id_trenes_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.trenes_id_trenes_seq OWNED BY public.trenes.id_tren;
          public               postgres    false    220            �            1259    16623    viajes_id_viaje_seq    SEQUENCE     �   CREATE SEQUENCE public.viajes_id_viaje_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 *   DROP SEQUENCE public.viajes_id_viaje_seq;
       public               postgres    false    225            �           0    0    viajes_id_viaje_seq    SEQUENCE OWNED BY     K   ALTER SEQUENCE public.viajes_id_viaje_seq OWNED BY public.viajes.id_viaje;
          public               postgres    false    224            �           2604    16760    conteo_pasajeros id    DEFAULT     z   ALTER TABLE ONLY public.conteo_pasajeros ALTER COLUMN id SET DEFAULT nextval('public.conteo_pasajeros_id_seq'::regclass);
 B   ALTER TABLE public.conteo_pasajeros ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    231    230    231            �           2604    16618    estaciones id_estacion    DEFAULT     �   ALTER TABLE ONLY public.estaciones ALTER COLUMN id_estacion SET DEFAULT nextval('public.estaciones_id_estacion_seq'::regclass);
 E   ALTER TABLE public.estaciones ALTER COLUMN id_estacion DROP DEFAULT;
       public               postgres    false    222    223    223            �           2604    16634    trayectos id_trayecto    DEFAULT     ~   ALTER TABLE ONLY public.trayectos ALTER COLUMN id_trayecto SET DEFAULT nextval('public.trayectos_id_trayecto_seq'::regclass);
 D   ALTER TABLE public.trayectos ALTER COLUMN id_trayecto DROP DEFAULT;
       public               postgres    false    227    226    227            �           2604    16609    trenes id_tren    DEFAULT     r   ALTER TABLE ONLY public.trenes ALTER COLUMN id_tren SET DEFAULT nextval('public.trenes_id_trenes_seq'::regclass);
 =   ALTER TABLE public.trenes ALTER COLUMN id_tren DROP DEFAULT;
       public               postgres    false    221    220    221            �           2604    16627    viajes id_viaje    DEFAULT     r   ALTER TABLE ONLY public.viajes ALTER COLUMN id_viaje SET DEFAULT nextval('public.viajes_id_viaje_seq'::regclass);
 >   ALTER TABLE public.viajes ALTER COLUMN id_viaje DROP DEFAULT;
       public               postgres    false    225    224    225            �          0    16757    conteo_pasajeros 
   TABLE DATA           =   COPY public.conteo_pasajeros (id, total, tiempo) FROM stdin;
    public               postgres    false    231   �@       �          0    16615 
   estaciones 
   TABLE DATA           D   COPY public.estaciones (id_estacion, nombre, direccion) FROM stdin;
    public               postgres    false    223   `A       �          0    16597 	   pasajeros 
   TABLE DATA           `   COPY public.pasajeros (n_documento, nombre, direccion_residencia, fecha_nacimiento) FROM stdin;
    public               postgres    false    219   �V       �          0    16631 	   trayectos 
   TABLE DATA           N   COPY public.trayectos (id_trayecto, id_estacion, id_tren, nombre) FROM stdin;
    public               postgres    false    227   �       �          0    16606    trenes 
   TABLE DATA           <   COPY public.trenes (id_tren, modelo, capacidad) FROM stdin;
    public               postgres    false    221   ��       �          0    16624    viajes 
   TABLE DATA           Q   COPY public.viajes (id_viaje, n_documento, id_trayecto, inicio, fin) FROM stdin;
    public               postgres    false    225   ��       �           0    0    conteo_pasajeros_id_seq    SEQUENCE SET     F   SELECT pg_catalog.setval('public.conteo_pasajeros_id_seq', 10, true);
          public               postgres    false    230            �           0    0    estaciones_id_estacion_seq    SEQUENCE SET     H   SELECT pg_catalog.setval('public.estaciones_id_estacion_seq', 1, true);
          public               postgres    false    222            �           0    0    trayectos_id_trayecto_seq    SEQUENCE SET     H   SELECT pg_catalog.setval('public.trayectos_id_trayecto_seq', 1, false);
          public               postgres    false    226            �           0    0    trenes_id_trenes_seq    SEQUENCE SET     C   SELECT pg_catalog.setval('public.trenes_id_trenes_seq', 1, false);
          public               postgres    false    220            �           0    0    viajes_id_viaje_seq    SEQUENCE SET     B   SELECT pg_catalog.setval('public.viajes_id_viaje_seq', 1, false);
          public               postgres    false    224            �           2606    16762 &   conteo_pasajeros conteo_pasajeros_pkey 
   CONSTRAINT     d   ALTER TABLE ONLY public.conteo_pasajeros
    ADD CONSTRAINT conteo_pasajeros_pkey PRIMARY KEY (id);
 P   ALTER TABLE ONLY public.conteo_pasajeros DROP CONSTRAINT conteo_pasajeros_pkey;
       public                 postgres    false    231            �           2606    16622    estaciones estacion_pkey 
   CONSTRAINT     _   ALTER TABLE ONLY public.estaciones
    ADD CONSTRAINT estacion_pkey PRIMARY KEY (id_estacion);
 B   ALTER TABLE ONLY public.estaciones DROP CONSTRAINT estacion_pkey;
       public                 postgres    false    223            �           2606    16603    pasajeros pasajeros_pkey 
   CONSTRAINT     _   ALTER TABLE ONLY public.pasajeros
    ADD CONSTRAINT pasajeros_pkey PRIMARY KEY (n_documento);
 B   ALTER TABLE ONLY public.pasajeros DROP CONSTRAINT pasajeros_pkey;
       public                 postgres    false    219            �           2606    16638    trayectos trayecto_pkey 
   CONSTRAINT     ^   ALTER TABLE ONLY public.trayectos
    ADD CONSTRAINT trayecto_pkey PRIMARY KEY (id_trayecto);
 A   ALTER TABLE ONLY public.trayectos DROP CONSTRAINT trayecto_pkey;
       public                 postgres    false    227            �           2606    16613    trenes trenes_pkey 
   CONSTRAINT     U   ALTER TABLE ONLY public.trenes
    ADD CONSTRAINT trenes_pkey PRIMARY KEY (id_tren);
 <   ALTER TABLE ONLY public.trenes DROP CONSTRAINT trenes_pkey;
       public                 postgres    false    221            �           2606    16629    viajes viajes_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.viajes
    ADD CONSTRAINT viajes_pkey PRIMARY KEY (id_viaje);
 <   ALTER TABLE ONLY public.viajes DROP CONSTRAINT viajes_pkey;
       public                 postgres    false    225            �           2620    16768    pasajeros mitrigger    TRIGGER     n   CREATE TRIGGER mitrigger AFTER INSERT ON public.pasajeros FOR EACH ROW EXECUTE FUNCTION public.importantpl();
 ,   DROP TRIGGER mitrigger ON public.pasajeros;
       public               postgres    false    219    233            �           2606    16689    trayectos taryecto_tren_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trayectos
    ADD CONSTRAINT taryecto_tren_fkey FOREIGN KEY (id_tren) REFERENCES public.trenes(id_tren) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID;
 F   ALTER TABLE ONLY public.trayectos DROP CONSTRAINT taryecto_tren_fkey;
       public               postgres    false    221    227    4841            �           2606    16684     trayectos trayecto_estacion_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.trayectos
    ADD CONSTRAINT trayecto_estacion_fkey FOREIGN KEY (id_estacion) REFERENCES public.estaciones(id_estacion) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID;
 J   ALTER TABLE ONLY public.trayectos DROP CONSTRAINT trayecto_estacion_fkey;
       public               postgres    false    4843    223    227            �           2606    16699    viajes viajes_pasajeros_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.viajes
    ADD CONSTRAINT viajes_pasajeros_fkey FOREIGN KEY (n_documento) REFERENCES public.pasajeros(n_documento) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID;
 F   ALTER TABLE ONLY public.viajes DROP CONSTRAINT viajes_pasajeros_fkey;
       public               postgres    false    4839    225    219            �           2606    16694    viajes viajes_trayectos_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.viajes
    ADD CONSTRAINT viajes_trayectos_fkey FOREIGN KEY (id_viaje) REFERENCES public.trayectos(id_trayecto) ON UPDATE CASCADE ON DELETE CASCADE NOT VALID;
 F   ALTER TABLE ONLY public.viajes DROP CONSTRAINT viajes_trayectos_fkey;
       public               postgres    false    4847    227    225            �           0    16708    despues_noche_mview    MATERIALIZED VIEW DATA     6   REFRESH MATERIALIZED VIEW public.despues_noche_mview;
          public               postgres    false    229    5015            �   h   x�=��B1�u����v-�_$��>��j��q�p�ž�M��8��(ӄ����y�k����FAQQ"�WԘ�L�����+1$��������C �      �      x��Kv�:����U`}�'���T�t��T��'P&�d�I�L�Mo����0r��D|�0�]��
nO�[FPR�Z<��ŗͰ��4>Vi���mؼ�mk��n� ��㰉���f�<L�mh���K@}w���n�,��h������E\�4��G`C�rp�zZ�^<l��n]^�쾌��iZ�1�O�Q<����g�v�ZxL�q�&�R�����2��8L1���[��p��W��"m�/�d�U��DF�s���.n9¨���UZ��i �'R�;���Ppއ��S���u����}Z��g��S6VIt2c��'����%�	����.��(��m�i
LLi���J:P�[���>��00:{������ �oqџ���O���m�l�2�arg��������qׇ��O���f�l�"�q���^G0�T#����16���T�����F+%�� �F�̬R5]�n�4�Wڪ����ɪ��rS���a�e�?9�
1�w�tWܥaI��᫪4|_10��wf��|�r��"�J���:�H�o	=VW�,\�@�qjE�a�.A��a�F��l���g�W����� �;jħ�:rM���F�f�[q�&X�)-�c̙�KR)�u#.;q�r��#R��R������L-.���j����������Zq��~����i�|�[cN�(�kq�o��e�K���ݪ��F��(����}i��.r+T��y�-���Lb��2Ǣ@�-\��O���k��l��Pr��ʈ�vxu�	�4�o��*,0V�����0�
E"`Zk�����Xe�t�$��Ƌ_���\�J{x�Ո>���:l^Y&�DejzO�C�A;q��H�%?�PH������2������2
���a�W�~�ӊe�Ih8ǏaE1:H�n�Ea򨌁{�v���6Oq�^��*c�lX�~���9w�W����8J��%y��H�t����ȣ�"΂���ކYk��#�-V���t'P;�F�X��dWW'���8���ٰ�yL�o��5|Nyנ����F�t�x(@��6�<��gD#�SX��_�;���R��%�7Fh��7�;��8��Y�zӊ��8F�)��+kf-|S7F|+��Ee��J�6�v��rd��<Xe���V)��Cq����
�E��8��2�7�8����]��3��VY_��D}"֑�9���Y��r�iv�_�SB���:�c9�` 	�q�d��p���X���қC}ɔ���:����;���i������7?7�rnӦ���n�8�t+�*g�|3�{�LW(��t��ӝ 7Jۀ7���B�������T�����#R��w�)�.��^̒_�	Nb�ud�9F#US�M7,�b~Y�Q��-��zdU#�.�9L?�	���p�U���)Sٴ�F|Sy���B����*ӗ;fͧ'�߆�7�����+C�J	�!���c.�?��ppz�뺶B��a�94�T����Gi���j"��d��Ǐ�X��X O	�~�6&��U[��f|�8W���t�SU+Iv��=�bN[ߊ���wJo2S�U�t7N!���1�(�f�����Hk��{��8*9 �֠��Qb�/Fj�����Z�֠}���7!�U,�n?@Lq�3䘹��h�b.��ե2��28����1�(-̴�x.��p���i�����A��z�k��< ��TN^�7+�gN�>��֦�g����[ RyW}�����AY�5��a����?ۑ��74���$+�q�A�a0Ա"�� Ʀ���#���!Q{�[�s�E�.E �G!R-�K���U���2�\���lprV��]|{��伡n����@���U���{~���k��r�!�!�18kښ��4F�'�p�<�M[�s%�[�|W��a�T/B���v�M��0��\���!7��4�<?{\�SKq�"^!���Y���R7/�(	P��Nd��/���"�4La,K�H��]�ש߯�V<���u�Z+�A��h z蹅m��5�r*�ڴ^��/$!�>��~��6b�e�"	ױ�T�5~(�1�� ~�	����~'xӠ��<�/)�xxfbW�a�a˛�1+~,;nb��!K
�'�/�t��|��<qD��2������:��cZP��ˈqm�W�_�ܱ[�B�@�C��U��d�w�!
xItS�vx2�j-�rح���]`RI�a�2^|X%��[A>�/�p����<�BB��}�@��C.���aP[Q��[��a�s�1��H�0y\e���J�Ĝ�����Z��]�}5��_aX��Z��k��hJ�D7�+�˚P~�HA��?�VaM{0���܄\H���&A|�Ɖr:b���0�%ʂ�~��F]����RP�{6R	�f��A��8~"kp�4ԯ1��m��g�=n�є���2�!G��̢�0p�FC��J,o�BÂ-�t�8�L����[rp����M%���7�)F�:��<�ׅ_N�x��H�ܱ���)��s�"��<Ј�:���KS�
xi�(9@~>�(,�܈e��H��R&]@P����T)x�S��bH���[zW=�i��-��0�_�Ó���8��������sQh8�/q[+�E�^�����.]n8�E���A�ig�\K���"��j^��o�8����㕴5���4E�����D*�-a�IH8�c�*qFsN�>$C�o:*�9�Z������q(Ǣ�	���qǲ(,��U�S���ݰ���Xv���Ȟ7"r�F����F|C���� �|H��}0�R�B0Z��!.cƊ�C~����e�&
Q�aJ�[Q���]�Ӏ��U��52\<�0$�����ψgô�#[]�S'ܶRb8R����A�2�~��8�����UN�ԕǋ�(8\��j�N�E:=O_����M-?L�e�� �م�Wn����p�"��;�i�b�w�Ȳ�Y�p<	P7�5b�he�ڌ4����XK��-����lj��,i�;>Y8��p��p�7B�%S�PPB�yh4tV|	sB�]؊<4P6'�iM#�KK8���0(���D�N���o�L�c�Ǌ4h��{�,m�]��C ϩ���p�-wH�-���,�F����h��}ږ�2�;�I��0�8�E�h��N�I=ǣ�8<R|�G�@���0�~�q�/���Vm��4s�N�8$��ܡq���t�mɤUp�'V��W^ً�!},i+M�����EO"����%��[��J��E?c]��k%n6��\�wd�h�l	I����ߌc�{��>d��א�	�.Pe:p�w�U}��2ג��MMɾi�i��
���9������u7N݂c��,|E�F��ـ��[A�CQY�0�b�j�����!\kt��æ%�X�g�ġ�4 ��J��%��>Ppx���eEA��4P�)��oZ�(4,|I���G�>�`�"?�n^-i�&�Sم��h�P��uB�P�F�$Z8Òt�J|9��Y��m�(���Ptl�#i�C�ʮlDV�����2�7n#�"Fm�gbG���@2�Y���K�g�Ӗ���1�cd^�q�-G�+�ZH�OUt26Oq������k��GY�RN��qW1�d�pT-y�A[��"a�XE��OUӠ��Ӗz��Ue5~s�OR��h�E�m#.��n�׃���ͼ�,�V!/Yn�S�ZC+�Ө>C��(��iH�A�iS'Ά�,�	y������".B�q{{�k�v:���rA��J����y�3�A4�~��it+�
PnrdaO���U��ᅣ�U`И�-e}����X�.fj��UZ-,8˯aJ�a����a��)wZ�kA]X)s�� J�}�@+)�lG�!
�o�!�U��OK�+����h]�}�V,S��A�~n�v^<@R`�8��F9�����z�V2B�R�й�,���V,�����Y��֡D팫ˈ�vk�m7�j~#���Q�SP4p����a����� G  ����ID��b`ֿ��p��|g�!
�u� �F��*wk�Ob�k��);5�i�M���!
	כ�˲j)
��(C
���81@��$�7/Od��$�ӿH��X)��2�ǮU�a�"��e[�X��[Z�����i�D���M��nx��B�j���v�e���e��Uȹ#��U5O,�!��t��Y]�:���:��N+��m!
�"�\�<��k��_V��!/�R�L}�ڊ����t/����B@'���m���#A(3�`|�a�&;�/��p��H�Kݻ�|��Ȯ*�<�a��k�ex���-Bai��"��D�F|	<:��pp�#>&�6�+�1��><D���X匘��ܤ(dȂ�vc���·$$F!a4��Y�_Els_��߉�WB<lJ��k�ħ�R�'(��K��<$�YOh�kq��r����DZ�|�� �q�#��Yh�H}�xs�q��0���0t��N!Th���o����Z�~ƷU��+~�O*�$F1d��s�{Z�V{q^h�y�F$#+��k�O�|�U��D�_�jZO�C���%C0"��|`I���?�I��U�����y���1��Mv�'�𘶴Z�Zk��D�kJ����ׂ$�o����ZK�1�������� �� �F�˰e8^��0��9������9�S���[}���Vz�`Ȳ˨ q�n��|>��F�C��n^
�h�S1��F*ߊ�H�S��ģ�}�U�?ʹ��扅��~��h�I�i��M���H�j���U?��mC�(�R~���{Q�_�+��]
i�|ϣ!�}F`��V�͝��|da�iqY��/��=�����^�TRģ���ِ ���Uq8؏��Z|�,�C<�"Ef���tP��\Ȣ���z�Q��+Y���� )?�����縏$�بT[P�'~V�3�NW+�k��<�z_)`��($�q��u4?����Od�����Qg��w)~)����ޒ�����I�f������UʼF�!��i����O�t0��qUf#�(2E���s�W����g�"��so'y0�h�]��\C1B�iqKJ=UK��Tp�:���q/���2d��SyTfhs��²�����A9t>�v�WVko��,���E�ֈ����@�*3��0p��#>�3��*0h�g^�.�8�dz_8��^A�lxU�,xؼ��2h�h�J7���7�0Z����S��䯔��~�'��$GqA���a�4�٩t]�Ҕ�ǫ���*	�0vS�)�-qK��S!�>N]>^�:�q��be�B��q
����w�W��,�f��ɪ�������6v      �      x�u}]s�H��s�W�m�z@
�G˒��G����L���H�@@Rӿ���U��و���1)��y2���T����Tw�y�^�ӡ�-mZn�n���7�I��+K��*S������U��?�,�ܵ}�m�<��T��oi�[殴���j���aT6�\O����3?k�oi�[j���:<7m[o>4��>��H�,߼����n�v����-I�W����~h���j<4���w��|�s��~m~U��~ �-�WV}�:���~:�A>^l�^���_��Ӳ��%��^�Oq޼kN�C�Ϧ��x�]��|[w�z�Ӹ��p|�m���^�l���A�u|��w��\�|yU��M�l>�O��*�ћ�m�\o>N��������7��U��g�7w�ӓ��|s]o�o;<r3�O�^�)^r�WC}:����n[��rs;��o�~�V|�R�zq�f��Tc��Sն?N�m�\�m�5����n_�2:lX��5��z��^y�������n�N�OM�?�[j�R�_��n���Ɵ7�s��{�<�A>Q�'��I��}W��֣r�,6�Հ���bѳ���*���xl��w,��Fe��A���`��g�g=p�>T�K�U/�|s�>V���yx�~N}?4v���GP9<����ί>�y��*-ջ~��ύ;n^�h����_nF��$&�U���pZ���թ�/�X�om���.]�]e��P��_��Jg�kw^��� �&�&.b�Kֽ���
��n���U\b�n����U��۪�����v=��*�-7��D�d~������l>5�C��j��܇�T���Ӎ�6���U����X��J����o8��?�`���)�Ҫ?��c����쇾�8�X�M����ƥk�3v�G�/=D�{:�u�.����4�'e���S��������T_e�zӝ�n��_`�e7wSw?�Y��,�fV\�Dݎ�z���q����ù�m�����+����J���թo�͗�۩m�J�y�7Gg}�ņ�镦�*��?�I�b��R�wu˕���K�s�`,��	�3}Ǚ�E�j�F<��KuϏu,���:�s��<�q����d���m�m+�j���/�8�����_�.:�Y���鴜�L��n�]�=�R|;4�S�(r ^��Z����_:o��O;erWn`U�\��8�W�E�J�Vݎ�������?���X�E��@�Z���i<�%��ͅ���8����W��#����e68����O�IX�7�O� �a�ʤ�=�:��O����x�;)��&�i��/�,^����L���'襣�-��qy��`Dh�ܕ��C��r��o�4��lO�w,G���׸1���aT��r3�ڔ�$���~l�o��-�~�
��M�pofce�Oa��U���5.C��yM�\8~�p�?�b
��	�2�z������/��C܁�p� U��}_E(W�8�m۬N�w��� ��Q�6��rc��[B�y��;��	�~��?�>��/h��m`_�熯֝��\�pn�=��$ޢ�g�'W����CuTx�/��pu��@0�ۘk�����LXz�(t�����?�zOＱ��Qo���;�)��C_?�2�6����9�C�L�{�-�`<uQ�W73l@�ኾ�-��8n�[5*�>��94����������B}�U�P�ͯ_p�*!��5�%>-��+Ks�;��������ؕy��î�-dg����9����`h���T�0]�hʚ�x�_�^�w�+���=󶏧s�E���&F�+�o���Tva�؞���X�Q�Hėn�:L�C#N��uW6S�q�6�kx;��g6w�ZNȏz�U)e=�i�>Wx��<_��#�6�X�u�N��Q��#�d��ӄj	}_*[{ms~�&^F�%�B�S÷���g\P$ղ+kտ��j�T��ɘ�O����R[��  ����u{`�b@V��X0�u���< h0��o@exl�g`�P\Y��ah��ey��?ͫ.P
8�Hh�Ò}�q'
,���>��7�H��P��뎮s�ׂ��aw���9;�R�LqUd���!0iwC��r�>��7�	~6	>����з��7~�y����/���\���bԾU�w@ȸ�����U��Ey�w'��A��o�X  �g�{����ȃ{Aް�V�i��z��S���$��X�����!+<&Ȯ�B}���3�Z�����������fo�Br{�q�ar���U�K���/8���#�R�t�Ö#8���V�4mEL�:�F���r��\�׹P_@���(�.DK.��
g�<�ʸ<ݼ��㶟A����>.�E<�N#L�}�;daI�|�캴�]|����7���$�d�Q��J
gW� 5uxY�I,9�"�]uhd^�L�\���zl$,rˣ7����QCg�G� �frL��[�z,
Lz3�{b���\�M�S��j�U?���AO����B	�ewj��8� >DxQ�ElH3%��D�"�������<��8����@#�
�0��%8`��/�
�����~�]�	ƶ�cM���n8�Cj�'<�|�>�We��~����ay"�_8$���a��J�R����x��^��m"��<?�7U��F}nƵ����,����e�U���W�/cg�Λ������:��rQVx�.�',	����A���N��港��Źw��,�-���
n�e삻�0fእ�Y+�Όoo�q>.��z�fw�?\�h��Bd[�V�;�C��{bOo Y=`�4IM�Q�2�N�G�)�H�d$MR��OH���Ɍ��T'�h�9�ʟ/��L��z�:^��.	�p
yDJz�4a �%�4�a��*���M3{Z�N��[��4�� �� �L��A���P��� �i�����nc�K��n��."�\��Ɨl��4�؞~�A�]��?7Q`*ֆY��PΒ8�+��"�!Ἧ��t&:�Zz��zd�J�v&φ���O#㠘C����pͰ�#�~X
A^� \�-i[��4�u�}��2E1� r���K�H���5HS�	���D��K���4����g3I}��7\�r0�����f�����HS&yOOX١���t�O���Oծ�M�a��F�=�q�_���*���l{�w��n�q`G\�/P��QeejVh,�|N�%��k�������3ܽ�.��0��c=���'Ӵ � ��F���� p}�Xe'�(��,FU\.������f$0^���]���K%	?
�}lpN`a��x]�Be	�"���k������ �H�0Ǵw�g8P�_��0�Ƚ�K�p4`ۚ�Dnީ���πq���;���`�?����ϗ>D��ᇛ�ɝ�`U���>O\��:$���Έ�c��
��*R�u[�Խ�P��L��G��h���N����G�8��K&�@�s�H�0��;��67��k��Ot��#�J�B�a�wq�2��o�H%�f��M72�����s!�l��4���Tk��A"`,@��L��1������xj`��P~h�d>x�V.a��J�8:U0n�#�OhG�����g�/ć8ׂٹ:S��p��NJ�˼�]���E�6�+��0��)�a����_��l�
�g�[c��N�����Y%0��gjx�g4+�"<z� 7C�0t#>�S��:J�wϊ��iKd1���ch ��U\�������XW@�$y���*L�` /��o���X��9��}+!O�.������[�0��E� #ʰ�$���OԻ��Eo�|"�/}7%%�wMN�<�I���z���p����� S��k�i�!{ຩ^}��m�w�g�R���0rݯ
g���7L 3,^B8I���޵��X���g��v0�c]G7�˱&,a2k�%O�a-.�D�����7���U����G��M�g����}Y_K7S0�=p|n��zU�\�@�a�
.��2�͝�T����n�n������'܌�@Z����N�!(��&�\/�tN/�p4�y�ؾ��<��w�@D4�D0��G�    �T� �4c�5�Z0Q� �}�=�4���<�i,:Ͼz�+Rf�`㎴+,�'����m���������d�ۜ�{�jKn�Ưp^�>����k��n~�\�D�ɸ �8a)�pwL3����@J^C�O�H�����>�/�r��l,�ŗ.=(� �1f��T/����fL�K.���._���<T?�O��b�S9���y	�8�0B���7�=V_1���	1Ij��ڱ ����r��Z^��M�gS�q\nsdm�ia������4�����J0\֯���ˊ�񵔓V�!a(Z�m�nn�N/]�V��Z�-����N�F� ���#c���N��[��鞈�SM��̖,u|�R�6n�E�|��|o����,C�~�T�������<V�D��hl������!��Z�B��3N�p�o�KYsI���Ԗ�.��Wf.�5�p?���l��|�"\k	�n��,��p�o~p�S��(M��rZ����dA_i�ei��/=\ű���M�L�&� X��Z*و�w�Z��D����V?u{����O�. �����n�v6�V�M<�c.�z��]��E,�;�/��"YC��O�88/L�<��#bZ�����Y8���#L_ˎ�|6�e.9E>Q����5��;�zRI�u�lÃX�q�q6%ɴp�u�Z`Ɨ�����ڥ��hq.����D�����ŰiN�72?�\�s��8@񉛉����Q�<��0w�RxL�gs)nȎ��"�(i�:3����?���,�֝OM�����R�)���|�s	Xh���6_��R��N+���%>�bñ��� #��������G!/�:�������m��y!�0���\H�h�Yț��:�L�Q�F2�x.�����ag�lRI� �v�����?��^�.I����x ��;<7Ro�O����k�O��ͱ�Gg�W6�爭#��. ٮTd/��*V�q6wO���=��̗���|x��nq"��G�� �]
�>�L�B���ᡂ������Y�f1��R� �*8��pՉY���U*�+Y_K9w⪷�i��v#��ǩ:���/�����󛱭��ȸOks,"=�R.������o��a2�M���a��ƅ4~.�K�O,����y�g�1�O�C8��?�o�:��hDo������CZ���;�&L|rVl���qB�e��^���ô�U0��๞��
�h�NJBl�'U�����gXe..��x�Z�n�s8�@��8���k�N���m��t��kuW��q�`N'f	����R&`����~̓�/��4),�̔Z\�,��,�Y�>IDq�pJe����Cb���c1���� ����,���O6���%��C�O�����1��/NC->K����0��qmr��
��4���_�@_c�{��u�t̸���&�۫���J*r�%����N�"e+�,Mab��}��7^�.�Aȅy��AD��Y�5��OT|{�6^R��M�R�ԧF��$r16(2 �k_��gpI���b�8���k�?ȡvl�O�$d��4W����@��gP��^L�J?��yfS˨s�	��a�LB��/��Dx~Y��0���F �l��ϊ���Ef��5 c�3�P+�D�DDkL�yS)B��&�Dm?*��T�0�jx�,Q��8��;����c�N��& "����+�-	)E��:�P�X?�2N��`��;aB��*5�-!|���'R>8���mS׳�Q�|�Ib��e��b1��L�"�S��!��	^,R��V��]����z ��1�\��ȁ74��%�%��}���U�����6�2�i�
�f�0(��ھ(;�G.t��ʀ͇�S��xyՋ�Y@�$k>��.3�$����`;��Y~J���{`��P�`Mo��q�adU�#p�ռ��R�t��>`�+b�Csd��{C�],��,Z�!�����h�{f�J�J��לe���0Xh�2���PI��jO+�.�)�Θ���F�] V&�Q����t\O��RC����n��Sf෇�;�#�T�L[u�{f8��wL�'��x�==��Hy�ĭL��0IQ���C/���ڜ%� j��{�|�V��#P:�"{x`!e�Jx*�����*��O�^�]kN%�2D�Iԇ��x@q�����՛ӊ��+�3�7�N<��n4)D����6[�	�!�
#�w����X�^L�p�2��j �x�U[!p��t��>��:|X�Xjbw�%�yʒB��rXe�²�ę��J���ύ0&X��ļ�I~C��f
�[}�NJ,)<K�8��p���P��z󯪛N��`kCb�E�Gl>�l��
Wv�#�|]���$�72�M=V4�,K���V�.g~�KlC�6~�m�(�o����?��P��mؠ&;�}w8j�"��F�Ĉ�|h���3�O�3�Ǯ"�j@��}l��:B���ܱX�Rj�R�O�pMr������O;��z���rx���n�E�"���Ґ
"ē��Mb�kV �	��z����_j�w�fr��������:p�$���B���m�8��,���)��9oX>c�;��j�ul`�z9���>���%� G�ج���u	�,�����տ������nG���:�)��kH��-p�����+��B2c.���V�f�φ���Fc�x˅��s��Z`�������G Kb�{F>��#o�}��0�TL� 
XK���J�e�`�xK@���2����98�������}Dx	�30�g�|��un��oʥ!Mw�"��������+}� �V�U7?���X8�п����Z_�ƻ�|"��Y��5iY̧�j�7F{�m&�J� �m�?����F�:��B�ZJ�y�6E��}/�,�0�ӄVR3��� *���<��u0$��ȝ�HB-��W���V�A8aI�	���4X��L(7#�T��ŕ��L�~5*s;�i����y�A_c�R��I�W���Abbl&o}�HZ+�G�>��]/����(Իv%�S��=Y��uMכ�Θ,h���� ̾]T��Q�B5+p��-��
  ]n���bxY]�w�	���6�Gg��PY�]��5��qa��
���aU��d�L�<Yb=/��N�|	W�"t�dN/T�?��i����\���"�('b�;�Tv,0E�r\��.@��NΓO0����m���DD6�$3g'�����5qɱ�!-86��y�\2n �k,>XK8�37��v��5t�T���2����r<]�޿�8���ٵF��f�ai�v�$��|ݴ㽰��At�Hm�i#>��dV���@`�n����~rę��.�Q&����WIľ�kxQy����^��aB��J��6�t���x�0��D2ک_��=�G~��?z_Wn�7��R�/څ軼��G,s����}Ú��#�^x��ܿ�.7������?bCq,^L���貀�w���q�p�wx&V�v�#e����5VM�N�< ������_����W���)r@�E:�����VnC�L'	@'��?�ꗲ�ξ�ӊ��)�R��I��Qm�/�#Ga��§���E���&|-S$ǐ/y�Ty���.�[�Va�ꄝS�dWk*Vx�����P?ՉQ��d�n��$q�e�e���Ó�|�[�<��I-a���]:a#��}C0"�/�	�r��R���z�R��E�	A��Wf��/���]��k�<(��IfVXjl�y�2s�:)I��5fKS�c�X��nF7Y��N��#��o��IS�4a�.("	�/Ƃ(�sGr��b^2���f��J3���o���J�@܆�د]�s`�XOo��Y�pe�-5���+	�p��J�0JҲ!P K��r.{p���m�i�1m!I�y:�����%>�}'�Z��jx�����p��{E.�$004�cuO�{��# u�	�g}�ȥ�_�g?V8q��ƭ�,x�,�AZ���@�ɛ�f`�cFx���y��c4���{\N���e��"�NWQ�+�zC    �x[�v&U��3��D�0�E�2fꅆt�o���s��s����el�`U�����Hp �2��c�C�@����y�%O3�G9�!�^��~ص,���z]�h�'r ���.ͪ�,N�`�H4���t�*(ׁ�� ��~��2��࿳j�A�=%�I�&�)�&f���;��y%Ed�h2�&y�j��yxh���rs&n�c\z���걕�0q��RgK?�:EI�Z���y��S;Bt�e����3���O!qk�	��9���s�ۖ�!f��@m��q!���V��O��K~&6�|��=m�*����f?i�N/�1��u����L��x-�
�ꑍQ�fi� ����B���I7'q�R�M�@(���0�I�M�SܳJx>iʂ޺1?�'�˫;I�{�����,[i�G�]%=����R]�rr��;H��}���dJQ:����j`Ǜ��3M���|/\�T2�4y��ڤd�H_Ll9g�^��+Z�T��a��@�;�g��髚�ܤ�D\�$S[�E�û#�ټ!~��&D�Q�+�����_�z�^�3&NJK$�--Pi��"ΐ9�C� 5�g���H<��5[$ih�a��@p��G!/��������n�9���<+��}�����|�B��ӥ�P FZ.�b�4���Z1�dƤs�e5G��}bb��g�,}]Kw_���y�Ѐ����gr2�O�;��jɐ�GҦ���X+��'9���P0ֹ�8{��s�%/�.�$��T�l��}Q��4�4#�kaz���a�ߖ�� �:�m���~4I/��n�u���� �7X�Ƽa6~y
\ҽ�5<"j�,�̡�K��a�`cw�/`�j)����z%�P��K�;i�lܧ�����+6��#y�d��<��wGp��cd�9*q���N�Yڦ��|�G&Jq��|��]�>��d����y��O��vn�Z�Y9�)��[gV nۣ���Q�t-֨��q�b�����9𰏟��6�j���upC&�	�I�eD�֪[.(#�����c���]�w��: V�����~;3٬Ϙb��c}NZ�1����H/5��@+ْ������X��v��K�#�]$0rGi��"�&-3��'dF�i�s�n�)C��f��ܑEŀ=��vT�!��|nCM�8E���=/`ͯ<�8^�B�?K������b��sƆ�H�����׭4C	�V�t���V���;y�Ţaߋ'̜]�s��k]y3���H�F�bvץ��y�u]�H�k�^�V�5�BKNt�|�+t�.���<�Ki���*'�di���c�{�9B��Ai��4�|�F�2��ZdlI�؟���#�/��_���8Κ�g�$����bî��3���W�LL<���^��(��0�%\�&��P���4m[���'��9��1fC�Z;�|���3b4F�lMʅ��B�4��J�����S���4��skYz ��L' ���m�L�57��◡3^;�ݱm�wp�4'��T��[S%� �����Ύp'���n|�b|o���5w�z+����"{��6$�d�s/x��CjG���0aa_�ɾf�#&�hB��Q@�otC�T&2�O�L�k�UIs1�Hs�ʵ$�����ر���"�6��M�3Q+�c���tEIqv'I�g	N��_�S:�I['k) �; ����Z� !�G�9�=K[R�n<����U��U�񲀳��W&- q���6��%��Kǿ�3Y�;�J�*`M��
K��8Te)M�B(%i�-���J$	s<�I�q��G��� X�W��>���:�$UwM��\�?Y�'�
{�1�M)��=���~���D�l�L�AC�$&H��=L"=�]P\:�����ɾO-���'b�{o�/w��lԴ��^�9�eC��0I6�J4�T�aT0��l��~
��b`���#��Ou�J��[���HZl�I�@wlXw,�a�]�\sSJ �IX�83/����]�@h�:�32IJH8���]JMJ����}2i����Dy�{����&�]��'Q�%Ed]K��쬬��:K��iF䆛��V��w���^����wLi��S�'���	���X���E��0`�Mj��Z�0��U,Rt+�eh+�n����7m�i/�֫�/�Ūo�! ���,�_K�A>���'���Ա��]Yq�|���>� ��н�n	W��uNLZFV��25�������e��껮�������fQHȢI⥥4>�����<�%E��<b2�5�8'&�Էa�4��C�:�K�i�~��T������QbD)x%�%.��~܆̨��a�gj�'kΥ��x�Mf�'��He��c[KϺ"�S�"	�Ր�;��|�+�%�ݮ��c���t(�P_}��ہ*IR/Ksh�Jt�
���J\Ya��xr'l�u�6���kWM�3��o��`�FW�(��?�$=��2���R�1�-��g�o������ë���
L�=���H=e�S�uQ2@hƆ1ILLm���י0�%�z١��i��7�^^��WV�2.����EsIJ�~/�I�����%�b"P^ؕ��S���U��'�>#K ��f�ER��m`]��aI?��{��V3�t���n��N�I�I!��d��p�T�\J�]xX��Y1�akGT��nd�N�RZʈ+�I��Kc��b=e�f�ƸxyH����=�O\pkr������/�FE���H]=3T�Ǯ���^)q�#� �i��K�^FΌ��s��39U�03�5�j�9��U^����$�V����c�Š���"�(�.#����z؎�;^s�wZ�4��*F�O��BJx���2���j�,CiȘ�Y��G�7l���:�����]HDS�E�����0�=.D�$ir2<G`L�����U����iߙ�A����Z��1�W���-�Ktyq�`�+Fi�� �]�	F��
���"I�\f4xХ��94b`� ��yV�Q�Bn�L�$�U.rw\��R��Tm�뉚o�+� ��I���MƧɭz�C�dp����hB�m�B��8�~�kJ�͐.Z�p�rd�Āf��}���Пש}�V����f��׭��ɹ�ߊ��,�}���wԅ��R�[��/�Y�|6U�X<V@��N�3��ge��d�	/�f�=��ëtϸ
�p$us�GI���Z����a��Yժg�����t[����qqf%�Y7"�U��ْ����n��;Ğ:���4"W�Z��P������#^#��a6m�v��-�zd��{��%�?�3����������.��)�3�z��I`KO�<����`jj��X���/rƹ GvQz��ud%A�J����3��]s��ӧ�$L�Lh.��?��M��AdH����ú�ʩ�u�>�꼹h�����,����p+i�5{��.���{���6�"��_O$,�ɨ7g�o�L�;���X
����^�Gi��5���U��T�$˝�2�l���tl
��2�<�����R�"]�-��a�`I�ziIO�w+'�a�,��aG"�8VE���fb�̼΃X�q���t����_-}�]	�\j�z��q��r[�H��}!l���K����7����a۝6��8�-9���iJ(�u&�{��2]	F���DN�3�x���uS�Od<���hIz�O~<WD��	B@�������Dv�!E,&���$\�|��y�5�@t�U�NԺ�!�0u���ʉ�?�#�`���d<3�_e�W$����3d��?	��\����\�)uSq"�#M�Ԭ�'5�^�X�)��c%�����4*B��_Cš.35WA���;�&�ٮU�dBbϔT \��N9m�X��u��X案hJ������?:�S�(��
�����dʷt�a��[t+Ϧ��Y �����Sp����F��V�s�m�=���.��;9���,`J�����Oݎ6 �oF�I����QrY���4:A��l8��� �s�<�;!���'й.�ť����8O2�����w/��p_.r+g*�y��;�U�
o��<9�A�38����yJt7�G���þ�z    �A�'B�� ���$�°�"��C^�ŧl����}�%�������3�s.��A;,O�832Mr�U����䭗��[�<aJB`*~�D��ɄO��]�0bhdeˠ�}dE�l<��4k/ 6S�Lh���ǰa�������;v�2�<M�e�~��m�q�C�5z�
�\�4S�O����݃����sjІ�5���Q���oRCs0��\>-��m�+��e����Ŀ^n�Rg��C�_��W���T$i���(vm�N\|rf��l�D�Cbi�{%��o�o�ǭ$:y�a��0ϥ�� ��.0�i��oN���(�@�T�O�$g�3\*�!��ma���O1�ߦ��d���1?�zq�Iȳ4��iG�G�Ɨ����W�ͳLHH|�� q�������Kd>���g,;���烝�U��^�Jq�����ſ#*�����ixV��h���c�**b�'�_1=u��?�$~�gV]�e`��l��Ⴕ�
O�\�����"�Oux/�)����)E+�jV^ǉ
�UF��Y��qD^�K��/���t��s3ʌ v�����=L.00s���$���Ql�$�گ4�4��sMzg��L�;r�x� Xv�����5��NrǞy)?ht�i9����i�Û{��qi&�S�����".�d�կ������)!F˄�����T�C�F���L9����V�v11�;CI7�"�(�M%1�J�o÷i{!���jdP�x=�\~�G8��IY���7�>IZ9��&�W��Rn3_����c�f���Lg9�Y�%	��2h0�&U�?� �N(jC��gIX�n{�����/=k�ɂ$*�J�p`k��1zxY�FCh��ϔ���LS~�?�t\=C2g�A��1=�(��%#~X�����$�]�}�	rl�JeRA�X��T6K���qpjK9�O�S�t�"�ƒK��jY�뷇�I�TZ������暜���t1�,��&���U�Q�c���tēD�r���H OH��@:1(��Т���{�.�0�'"��5!bb�P�_`i�B�)�/BV��.���v�8x��\�?���O gj3���E@���&號�P����p�wv!��$�U3�a�×�[I��d�~�yUR�b�S���>U��6�uv~	V�-nD��j>�l�b3Z��t6D%y����To�
E���YͿ�U��p��R�3n��K�#��l�Z�-|�d�$�҇I�F��,ƹN�!���44o��@��˗ְ����r��$"��
l[�1�Q�{lF�;.��s�Ä�^D�Y�@|�i�Vt�"�Y�d���g#32h��®������q�e�zG��Ϳ��H2�$p.	�	��dXJ6�Ꞷ�Q�l��cCX��"�zT�0���}5; �L U�֩/��x#3�x���e�x5lIK�l��؝�Ӹ��AJBESE�W�G����]�����v�:��#p�o�Ǘs� ��� ��ڼ9F%
�X������ɘf��ޭ�3�"T)sIqQ�'{���&Y�����0�S�ɲQ�j)��#m����jhR���c.l�j{2�L��k�y����i?�#"�}���*����B�BH=�t�q�Ah�_j�Y���d&ՙ5G�����{o'䫜/Q;�G��PaK*�^.�����b9�Q�nѥ.��/z�}F)g$�6L���Pd�����P��I`���ݞy�q<�t��r�5��8Kr0w$b��PU��b'i>���y�.�qgdN���Ej�z�}�>���j ��ƹ~�L�l&ZN'��:�d�2�ѹ�0����:����>�U?T���B!��՜�x%b�='���ۯ{APbhBC���h�\�eh�E��g�$��H�7�A�4/v��k/ۢ�
t����2�JVt�3��3�8
�eFG�
9�*�2Ȣ�y�9��}:P�C6�_b���x�I~�H� �ʫ��T�oU47D.����C%���-){;^\�,��祠보\	��@*��E|?�sx�B�k�U�հ�m͹:f5���O���l�PRd�Q�Kp��,⑥B�H\���s0�犯��e(�Yj�Ml��i:Hi�CO8������I��EH�<��g^��w]%�J�BR�'�|�>���|Q�*���қ��Հ����e�Ze.p�l"�#u-!p���R�S���b��R|6�!��c�#�Η��>����B�?�i'��!%' ��=�n���8r��K��wgK�s��Y�6�]�M��]�g�g���-qd�M&o%�{�j�<	�҂�����j�ăf�v^�Yd���$:T�+�7dAsz:�����D��m�>*�������e��Y�4��6�m�Y�[�#K�a��û�y-��x�z��qa�R��2�����i ���)R�4�����K���M9G��T<c!�ݷͩa"-���H�a����37��8��M�:��S�(4���L�?k��&�޼)�s���}�z���_�"�"|6��a�û��PS��`q~�H��$&!����u�)��~�����c˝�6�X� ;�O\�i��Ģ��h>3Q��f"��q�C� 7q����33"ç��X������~�Ek�6��~K��^&�ѡ���=�e�b�1�Ei�{����/�Z?l�B����y�tA��f�[.f3�Us�9Ƃ�_�|�_��#���w�����v�2�,�@{.^yn"�E�<µ~�#�d �M?�X�jJU�?ƫj��Kgj�5e��¯g z�e�ϚD}A��h��4�j]�p���e6S�CM֦��q�x��n����=:,.wv�R��PJ�k5}n�����6��%���\�Zm�?G�5��Hc�%@߷������K�;\!(��͆���W��W��9zj83�M}Qv��Q��T�m�ZB�K����r��y`�$=,aV�e���0	|E���zEx������x�M*�I��*?el.���<e��+�䈎�B��Qf���*��D��R��a9v'6��{Ƚ_����������Ȥ-�s��l�7�������ko��yNT�ʢe��.�%@n^��[dǣ�st�)�dvh��\�DT
����X���M��[CM�-�$��Y�$��3��LE	-vj��,�AY�`�W�#=YW*熳KSJ<')�&��e�S5AΈ�wć��d),��'���9J�`�aI(�k��fI �"�H�ֆ�#{���H��/�Э��	}67�o������a
+�����
�)���\��M�nO���t~?�.�����斪:>�"'�V�?�=?�km�B]7\bVq %�4qzY$��A��~�q��RP$����K\[�P��9O��qpdi<�a&/v.��^��L��^4��H�,N�]��
�ܒ�E�,��\}�K���@�Ϫ�����[ld�W��kQ�����2"r�s������W��g�ӵ+(C�Z%w���8{5�W�,Y�(k��_k���N�I�N4��#�3}��Hk���}l�����&��4'�\�VYfĆn<m����&_.=?��=k����Ls�G�V�c=] |Z[0�0>�5�p�e��]���˂d�-8�h �������Z��!�a;��o�4�aez��Q�~Vǰ���^Ky0F8��Up�*�D�^�I�Q�p�1��A�:#gB_�-�,`7�X���k���TFG�0�<�2��452��d�B3~}
�gD`���H8�{?��AA�`%g���q��B^J�E��75���aWSLc[T�T��M�@� Nq�N��3��b�R}:�����#�0�N˼��m����e�ވY��� t��m���E��Nx};)�e��N\F�^��26m���:qV-'���7^0xel�uZ����t���)��h�a��:#�#2�~�S�Hެc��co�u9��;�SưF�u�ˋS��1�(�I��Ke�r�p�'�ҊV,��шc�p�9Yk#uN�x����'�K$3}��q�J���pl��C�؈�@JǕ�G*� [&���~�2 x�R���� h[��7�    \ݓf˩5�er"K��-��x���$h�.�����č-��������Q2i�jC����Ϙ��|_�F6�b^~h�-=���I��Oל���4�F��\�/�ۖ�W��vXU���u&�s5����e���q,-yf�j��]>�����[���1���Kq��R}����3;=~��,ó"_*F�E�pf�.Ey^&7����@r�`[�$�$^��^�lJ̔�<�*8-�F�����E�B���iĥYH}��������~�(�'�e�?��A��!�/��?��_�fe��Pl3+({_L_����0\��xl���E⇌�W�S�ݙ�O���]�$�Oۅ�[zN�����~��V"]*AI$u�F=NQv8�2��>t5GK�`H=�y�2IP�%��'���F�n�_��D.�H����|G_�هvA��FcT�+��j�#!�����ER.��Og^�t�^��a]3/�>YH��Q���h%��)H�����e�_�L��x*��7O;�N�p����"��;!)�v�02ɺ�y�kDU��_}>ӻ��e��y��yl�+�B��m�rc��E�'��U�2dP�4V����ԗB'�L�u���nR1�ɺ����l��EMf���̼� Ҝ�7�8Q�b����YJ*�3�^}��&੹G"�j�M(V��B<�ӣҜ���D�G��y4��:�7��|�t����D�Y����G�#7}��D�%��jd��ד�C�����<���9 R&�I��	�M�٫��bN!"�|���~�����$]Z�#��yn~�y�j�U#r{�^�Z2����{��0���HV
s�ƫNb!uG4����aP���C1:��Ǿ��\-�ia�o91�c��lv+�����B��N#?�s/_G�E�'&�/X>�Yh���H���Q��w������[���&�MJ/z���"�!�դ�9ǵ �S�`�O�M	��`�=U�˩9�Y�����г�sd*���z^l$/4gÍ�
+S�:Z )��Y�eP9+��1�F�N�Xf2�q�i�~��z� }5d�!�	\��~�(�&��� ����r��W�%Y�ϭn;�;�dԄ��A�a�H�W�!Q_�,::��(g�a���\�g/�^�Tw�~��J˴ԙ�����*��SJ�.���H}"�]���wru�Lz��:sH���E��7�8�ƒ�P�%�q`��ań���Đ�#�pN|�9�=���Cus����]�����Ŕ�p���{�}����j��z���A9��E�p��H�sF�s��z�z��;<l�NJ�_���``V�P+zAd�9�Tw�6_��ZXl���f���0Z�\�7$.֕h6W�I�U'�#����x��g��?:���	���Z��D��/T,9����9��/,B�}�>ġ�|�wK-X����ާl.&֤��Ja�{��yD�{yW���E&+挕��jEe���G���"/Y!�W�+]F�KΫt�6	�KJT�pܓ(#Jb��<zdi�)I��*-������U'aDYa3�Ϛ�
?�%��Ϣ������ey�$o .�Y4��e8�B����� 7J-{ec�N�:�s�e�`�
������ϔk���#�OO����}�V#�n���9p$;S��rl�̠�7x�[|��d9�W}��2�J�6�j��*���f�J�"�5��1+>�\X/��i8ԩ4����%:�[}>��E�`	��]h��qY�AE���@�C}_m>�$#l�i�~Ο��`�V|��oM&$+�=�� ��q��p���-
�{���#CI
����z�o�D�.��+D�ʆ���ցx��
f�z�z�'懬�`�+T�\\V�4D���q$� ������Aǯ�lG�g��c�Zi����Jg)��0��(��y�b�O�c��*�n��,JQ'|�C����̣���^��D��gd�a�ٙE�,9l��՘JQM�WR* l~���R�O
TK�&|e��2?�%&JyHC�ڮt�jK���@��T��>��I!+�H�R����p��U F��3��BcO����T�>n�k7�.52���Ep�	&���:��S��-�U ?�p�  S��L2F�����Ϻ=�����N�\�/�W$��J􄃵���h��v����4ItФ��,��eG��bGdX�Iܭ����UH
R��Jۘ��N�>��Q�:<��a oQz��V$&n�\���T�/��G2���X"��_��?D� �A�	ԣV֥���p%�"D��4ɢ4��i)����<�T�?T�����4I��n`����+��N4�9ey%^���e������o��3�>P�����᫉sz�TA⥕$eۢ��(M�l.v��{�W��7�%��-9'7��w �\܇���p�%@i�����a�*�X3>2|'S���<�#cҵ�[������P�Fj�Eё�9)N�sԳ��I�g,Oh�aCX�?Ezv��Yh�&NH -��a�U�!��"(*sY*�y�� ��-��C%�%����[�����_��Z�[:\�$J�ZW0���"1Y��&Â��ă�8��������K���� D�͙�/��X�"�2.M璩T��ÖN/�n��֠��5s)"v�����09���*��n�(]��ԛ��\> ��c�u�y�2��9���c3H1��8"P�(�`å9[�ɑ}�����r��1�+�x~R+yy_��L��|4g�u�ui�����m���?l�.�(��E����)�5��\���GeH�9�x�^!i�	���d�_ԝ��QFb�"۩�ĮZ���l.K�ɣ��M�_#��q��2Iv��IzrY$���7�K���c%��&��\ƨZ�l}oHb9�Ջ�,	(�{����gEX+�R�5��Ŝ<��� '����pz�v:ʘ�(��ai`_�,Nޖ�C=v�k++a�U`�CN:k�h�� ��j�Ӌٰ��I���2'�s����쁠�"�i�[.�2��4�m���05�zNn�	A�Œ�Y��,C��Q�m������0������~wE����lͬ�>y(M9��0X�o�ʵY�_������u��i �Zh����ڳ	b�N��9���euqo���lq���s�W�\��J�����ҧ���Ns����7�U/;Q5΋y�ڒfp�(�t�0����Ķ:�b��v�p�ݣ�rj�
.��g��ġBN�;�s{����$�G81'� �����:\�Gj�Ҕz��1���<W&�q�"�<zVg�d���^s����f��I���{=ξ,T(gH��d�c[�^i&[U�/��\aQ��~ ��a%-�O��䄯���9�F(�H�Ng�๱����X	��V���u�.f�&6ԩ�)�碳+eD����k�h��9`=� �����I��Ea�P�v��g������f+��@��h������pP�~磌��1+�@<�sX�6@�����ݬL/��ʂ�@�v�ԯ���v���)/�ib%��>G�{������}���T�\�(.-�Sju(�M����W����&Dd.��hX����ȑ(v=�h�f!��r��kbB ���GI� �0���P�s�Uk�����FJB����(mW��ͮg��}�� `���Ӛ�k�ĸ��v�����5���)�D��j|Cՠ�hr����az�M�����d�m�<<���es�k橹��s6�$k\_��~�.�t	��^�%�@RvV"�Gs�����7Vao�GsV��\�މ�&]]M�򖗒t6W߄���}�J�%�ד�<�T�^ΒK�'��!u�ɓ2�Vj$�Θ����p�f��Pk��?8���Ɵ�D�Gf#�9Q�9s�j�ĩ�$du�k��^��ns;^p�����`5�O9>�0:Fu"3f_<o?_m�h��s� ^b�,�0x!'���� �/���˄啴����#��lދ�9UI8R~�Ԟ3��O�,6���6�jҗͭZ=�8�B�J��f��2��E8R�!�+׼k��t��,�M6��ͼ�I'l�Yh��*Yi�>�X�s�c�"i*|�W/	��A (  �Q���C]�2��qn��(j���'���O��`f�~�m�1ƜK�#q�y�I��đ��`wpy]`4��Lx;VE����y(�o~xs:��9:�T�\�-ø ��&���e� ��p�E������Ӝ~���ח��^fhy$���L�}5��c&������tVH��X�d,.��ַ�U�~��2������G��nӂ�(�~���,�� ԛ}?�Qq�{vM�p���C^&JD4�j��t�Z�)'}�i�R(r%�ѽ`��ļ	�y"~_wK )RWf�G�F_�?�:������s�øfF��)5��^��*��i�r�Q�X��O��C"�/�o����3Y=��ŭ�"<�|��]�� ��J�a�ϸ!S+�2<p���bu��KWp"^b��ޛutĘ0�0�Z�N�C�Jw�Ue�d'.ƴ;��WΒ<շ��ܩ�z�ß�Ec��L8�DR�S�Gt7��T-.^]�N�������R�i�l+���8
Q�*�T����!=��'I@~��>0ee����﫫��b��      �      x��|�n#I��s�WD�L��.��/�� �k�T�ե���$Cd433x�"5�3>c̃��0΁_|�b���ڑIU0P�U�LfF�ؗ����~��q/�ץ.�F�te�{T'��d��,S�X�k[U�Z��,̏�~c����ra+��rS��%��MSv�M�2S�G��zFA�ă��=�ک�QZ]�"��q��~��7R���9�i�������D=:�]�F�|��}��^�W�dUcj'�2e�A$Q0m�9�fJut�T�?��AE��\i�ٵV�S���UZ��ڸ|����`�r�\�7�K�`�kjLy��Z�Y��*�UQ����=�Z$��h0�r,���\�V�	��h�k	��mi�������;]XW��&!7��n�b�L���X]��+�F��_m�U��{���9�ݭ��<wku�
.��
1�A����^nL�Ju��z�c�� ������fbb��l�,����ί��Q*s��:��Gu�����®7u�gEq?���0��Լ����
^����b������>�	4e�\��
�� �Ư����n��m��d:�Qp���Xc;��{��љ-�$�ʪ��5��JC� ��P�ƴ��M�F5cnt��&~�9~�t6x�:-L�WW|d�w�qk_�r���ɹ��&_�2�J�����d�m�C�U� W*�^��`Za��\��W�&J}Tw��d����}� p�g��p	�+��!��x}}2�
��hl��]�(��И$8k��F��v�e+�j���H����w�?�F=��MY5V,��b�w;ʢj-Rd�����举�^�}���rS[��G�9�0-��7ܨ3[��q �c�7{��3]�"&�c�߼�[�1z�])K�K�2X �8͠�XK�ڕ^�Z�����\t����?��]��Ý��̺O��)}ȵ)�3L]�7�2J����^n����ς2�nם����V�O6��)��h�@yv��\�xA�xN(�ثD�UH5���׵{����-��:�tP��iiB�+�:8+��
qD26���e���3ڂ�2���W��'%�xe���{��X�7��[%^^s����
�l�]������d,3��[^z�G~1�e�zB�������B�C+5=����(1�x�N�:�Y,'��	���R�]��ôO���9�?�h�F�P��a往�p�{u��fD�o[ؐ�b��[��f�se��_�%�����}a
�ǜ�d������x}�0��!�����Ŕ���Լt�a�c�%���D�PG��~a�%��60��뿿a���I�:G8t�~:��N���;Xg�P�K)��j�!z/�~4�$��Y���Լ!����K��6�p  5��^ޔ�(�q�!}��AkB�ׇ��{|����s���e�����
��������!�w%c�^����o�9��=�#Ba����@b_���:w]�O�@Dt��g]x�F�rL��V����jl������G=�����g�p�p,�1}���.�o-�I�E؃���!V�c;ב�� �¡�% ��� �@3v%|}8L��l'��֚%�f�GF��ڋ-���_����*�
b��j6:b�)��l����*.>�y�
�DQ�"uӨ&< ���c}���X��6/���i�L?��0R�slpž������LQ{W#��zb�X��Ṁ�:�bGj�,�:Aw�uXJ'ŏ�r��Yv��|�B��5|c7�T!f7+��%�3z���`g�4@�+�a�+�7��`�+�u�-��Op�7�_��TZ �z��G�(ψ.�P~T�O��h�=f,
&��cC1b��\��x�U�*3��
|�/}q�\��[T�S�s�7M�P���>fe@���v��7�=�R�a<�Ŵ�IQoJ�s��^7��y	/g��v↷��8"M�����~�@��le%��/���@�w��Cx�h(Xx�s� �li��r4Q��\σ.[ C�#�����Bc# �rCs���z�j8p�P ڢ�5Y�@s��p��x( �wҬJ��;��pL�-��L#�@+N�3nf0�s���+���U���Ho�O���2N��#��B�����=n$2�m��PRu	(ۋ|�N(��?i � ��@�� ����#U��������T�pp"���r\ê63�쀏 ΃��'�n�*��!�y8��������8@���-:X��w`���6ޫ����Ld!k�-�{�c	/�E@��O��Ƭ���H ��VU
��} F���)�7��d��c4Ї#��)A�^�Ǌ�Ƣ����{02x�3�'n<���5�B�=�R�-�5K�@��OJ3�6��OM�K@�-� ��ݿH��#8&�\\.5Z����>E�?5� j��WG�\c��p�Ĵ)�i	��]�V���"l�����������}ʳG�;k�a�5J��A���j����x�T��q�2zo����~��Ơ�=��\��/eĀI,^g������Wp��<5 :O�?�v��զՇ%�M��>Ͱ.���ӓ��y҃FP�����d�#�ڐ|����}I7�[���4�]KT0�[h�L��b�fW;2�8Qx�cB_��'L{������_����$hJ[V�;uD���׿�ܭr��<�%��V�,����$!��21H��\��ր��Lcݐ@�*rM�����B"�(��<���1AQ�{�;D��;"�h	ѕ��Јʍ/~&��+2q
H.�InB(�E�p$�K�R���6|a,J%^�}'��H��|�O�.�SJ^����Ɉ�0�M٬�Z��q�����Y�,��n�Y\^�[�_pp�N���	X���S
/�y^��>;�|��l�a��*R�����8���=/����h�����_|�p�܃XF)�� پ�r�G�w[� "+Z5�-Xx���S�n��gK��n���li�iO�5��� ��=��O���,��WV{s�C��Y1k13 }X0p��_��6�c�H�.��V�YC�N�&)M1y#,�������:M�Cη�y �T(���䢓�@"�	�E�����`�r�I�T���m�6&0:h��?[�=����e;pY l&���h1x���>H��`8R0Ba5Y0$ʉ�._���Z��kN�	�ZZl	����/���l�l,�N��A���b��G��ώ߻�{��s�-Ѡ�a�Ot�%�s�C��F��Z!򧒾�z?�	_ /h�蓿��#Aw�����A��>�)p��� (U(5�sz%��w�}2��@���,-H�	0$h��n�V� �TH ��rGAщa��τ�儬����%����N��oӣi�4�����0� ���Z��:0�	�_�_�Ϗ1��̱3��;VCut���a��j����!��Caf�P�\�jJ1�g�Y��k�ͻ/Fg�̎h��bwFb#�}�G�E�tȭ�	�x�=�������j�Փ�+^�������!���@;��/��A֌����;ǐ�o�ӆɼ��>x�@�L��o�2�j�!�$�����<J�ߚ�ei��_6کjcպQ`f��盠B�p��Bs�=A4p�2e���2�d�T�\3�zY���������&'��ԯΣ��H��:�xmW��m��5`�1py^6L��� ��)"�n�3�	WRʑ�DGP~���>�8�4��	oܚ����C�����8���Ht�����7�ֺi����O{^ 0���%���!�(0�|�h��%V�tT�O���ڮ4���D2)�8���dgk]83U�����L����@}ڷp�JQo �0���+��0
u����\���j	�ç�k!(6p��_TT@���r͌��>>h%  ��É��2muHw��?��זQ�6�/�%%��ބ�'�
&!�ù,�3iX������S�Ѐw_�5�Ǥ������}1���������P8�5FW�0���0z%�""��T�E����-*w$��2�5Y�	��b�K?-IX    j���qqL�<"y�uMC�6v����Ԑxԓ�}j2� (jS1�_μ���,�����gi�Q��]���)�!`m\g	�s)�QO8mYo��Y`��ԥ_tI��ob���70�K�~��M���4���.�h�0��0=W���;������0B��p7&�XBlK~���W�t�8$Ւ�0k��B�Z�$�@t濖/��kut�'&���w~1�	/��(O�2+����I���.�[�E�#�$&�@������3�H�m8w-h��[�D�c���g��s���E��T<y�e݋�=D�� ��wD�C(3*��I?9.m@��}ݏxF���w r _\�-�V�/B���U�r����������.�\ȖF��Vܙ�V �$&h��to��� ���w��h��{嵂��k(��<9$�92�p%Ŧ����a�Ş�9�ӗ����K�2gTl�oq�!�8��^ò��R!��&F%i�z�1�� ��@Җ�4':�M���9%�TAH7M�潱~B�����e9� �a�#�����;�zz��F)�+�,�R�N��m�,�Ҟ@�,s4�]Ӹ��6쳨mFL�Q��o�o��¾��!Q\�:6xG_�Kd�]�&䪈f��CJ��
T���5a���"�FA�~�J� y��$����@�6�N2m�C �J2_��`�R�����^r(���������0��; �".L]��y�%f:�u���2X�r�@�U��7��v��c~�p�R�,�x_Ĥ¤R;��R��Գ�\�v]ŗ	��l�����&俗���%����" ��8�:�9Y�MlB|b��gRd�C�`&>��W���ӕ_!����3rS� ��ˀ)��P�IM����n�@������'�ܗ�y�u���i�`��ޓ~1Vjf�^��f�~l)	���.T��A�����XXd%�<�{%�e���X����3f���@$8x�3�g)Ȟ�!k,y�fZ�^%u)���`��k�FWb�/�K]ݛ~�? x��!pF���F� 0�{��}�o Y qEW�0�� S�Z�f��[|T�[sB(&�}��>�M0����Nfؗ� ��^�t���}/�w�7p=��օy���7 ��'��kf����7>!X�I+��d�wK�\naj!��O��eT���'
���-��bl�LkL?���_���M$��8�w�r�*���xM��5iֵ���a��N�J�\^�;�qL\�׋������֣�E��+���ұCj/|�/Y2o	�Ԯ�]���p�I..������F��#4��3����X��ʐ)���˶ 	
�Jq���ПL�Ub��õ�W
��$�2���fM��[�DUČ��ͪ��L>$��DAZ	Ou{�1CfI��@�� o2��a�/uƀK�ڒ�%��x�s��Kq>x�a�y�XjXpn+�	NJ�����v��e*�ubkמ�b���V��6��&�
^��54d� ���$Dt��D��YBd��S�L��������笶@Y5؋|a~�p����{�ge��Yw湴�A>������93��2�����RTE��߬tS�1!A,Ű��<��]��\ӗU�*[/�ʘ�Z��b�ێ5����<��<�B��G笿c��YR:�{� s�^$q���#�S����Dg��c���5�HX����y��`I�Y1��P\m��{�`� ��E��al���EZ�ږ#�����o(V��O�6R	+������k�"���� �9�<�t� �mRN-3��d�E����l����̟�bHI���7`t��H�L�y�6�0)틃�Ϸ$�1������{�|�@� s���Č�1�N*�� 󞩂�[G��(fT)5,�����)V,�g庂�g���c1ǂ�$;�JF�F:�HsV#f�R@�V則1uV+���) B@MVj�(�u���E*h�c�-��feCf3cUD=&��4���e�q��T��x���(mUT���e}�'ew;�s�̋��c��k��5�ZK�����a��\�}M#&KH��V�F��/�p&��m�������0Ӄ�q�/�1��P�����!��b��|��m�
���__�wŦ(ˁB�i[�m0�DR��6�z���-���`!M���Gs8���_��e�@Z��P?T ،W����|ǲ���3�K\dd��A쿂Y �a�c�L�X�N�ix�G���'�1[V��\+�ؠC��}v����VK]$9z*�o	��\��� +��R��q[���D�^ډ_��+�rOl��	k��(0�F���g|�@J������-�JS���~�>6���AP��x�wIش�dV��o����S��R��|�-��f��%�X:��|�g���٘#.D�l ��r�V$`!�e�<���"Hڗ�,�-�_xs�M���E��brO��le^��$\�j�.M'm �{]� f�_��V�^�zՖ1c�,�$N�%��´�]���3����M��~��H`$4���9h���X��t����f�o�X��3T
K-)ᡯ^�Q�N�B�l�A0z��)���WCWi���ejm\��[�+��<e���?H���Y���.`B ˽47�#��XH
K�w�_W�C�gL��*=�����kGL7��$��|ͺhrl�˼9}<S�l���acO�=7Ҵ���qz��`iX�Yfl���u�=��)!&&O p����{P��􏟗f��5b^����Ā�u�@S:�<�1ā ѰGA��W]�ŵ>��pl�����k=��\�{_]��X*\�%$a1�'�q�V���b�����Rиj�KG���LrR�,_L�d�1f2�'�ʃK;L�2c��¢�m/�C�kO�_�Q.�<M�M*3����]�1|��O[1�� F,_�tmXכ�5<0ơ�Xa ���,��zB���/[�1��g���U�i)-�1�%��\�C������L�����XIk���Ԛ��|�MK��kX�X�;@LNZhpb�.������E2�����zQ�ǘê��?�)m�,����0N�o��4U���h�<s�6�2�P�*l��4�f�$�-�G+��fc���g|�����Z�w,�N�D��K���¤ד�-��(��&�HNLJvs�hj�KkX+'5 n��/�/�̾Q����SP�v����7B��72I/������H7��:��G&�d�"ޙH�z��.-ߞJ����2��prl��>,���T��S奶y��d�r���
x���w px$�UU�.+v��3�Nq�֯����-����ÆS!@W<E�;�8ؗ"#060.�B��H�TR���1�p��|�w�)�Kr�s��0{3�M�eW��4�&,����4�@f����u��֕|=��d��z�V�KX��yi�Ki��bZ�n�C����y�=�^_�=̬t���;���������B��������>�R��h<���OX+n��pDN_-��R�̷ ��a3�.E�B!��%
���o���X�.%�&,3a�]���G��q�d$��m�e�]���"��(�%��3�N2GI��#��:��)�P�A �֬��e����3�3u�T�)d��q鄂��S,��.l^ ��a(�t&em�fP����� �S��E�~2�O/e�|P�?�Vȯ�=�U��Hh;4��{����KE��d�7}趐a\N� �;�n|L���y�������;�/�08��1��,at�C8	DȔ#s����az�%J��?��?��[�HEN:M��ƭ��O��+�١���>����;v#����qD�o6hO��l��A��|[��-u�>����:mj�M:F�)���>�k�M���_�L�xP���O����Lu�Gv����C�l�!agO4����7I<���ѵD����=Ul�a��DHU�X8�<��A��#�e��Y�a�>!�'��F#i���u`�RN� b  Lp)�:pLSTF����/+�������Ւ�ؔܣ*���v�^iQ���ې/���f>�m(|���5!5��/����c���L�e,Yo���/�6&�Ք�6��:##:?�	��j�'�2�c�}��p�6��ߣ	�Ǉ�	��q�m��:�QsԥP��H��r�����⨒r�Ln�:RpF�#@MI��c��2I��j���Zn��ub�ê<,¾�d )'����9/�q��r�����E�JvJZ�yC�9h�Q�c��c[�=��t��w�Rn�2���)>�]�����1����PA|�M�j�F� Ci�ՁfR
��ƚW|�-�sڶ����g�z�[�Ɍފp1�3���/?>�|�'���A@B!���#�	�^�<�7R���p$(��t�S,�����T�TΤY7U��&,��S�f����+��]�G_��H�z�_ pQ�����V���,�����R�� |%;���������{k$���{�8�f��]��O�=.>P	q!9�*���_��m~u�2��#I_���s�����w�����h%�3��Z�3]����;BP�,25+	žj�`:�����X,�-�|Y��țm����?���'�-�:͒K��M�оz	���Ū�kp�g�����`�� �jC\1�Wa2�m?��d�}�Kc���1d ^UVC�0`޺P������h��$�b07Y�@���l���V�j�HN	6�@Q��L���&�ƥXr@��4��xa~��>���E�b�9�H�}����\$�����6y�R$+tQ�G���<6B�QK�&gT�+��܌��q���<D����%�ԓ�ekÔmeq{4z� td�롴�ҩ�t�h�0I
;~�n�׉�-ڄ�g�|��^�M`��B=�^�mT�:S�5�቉ �"����" ��ng�z<ږ1�\��nJ2�!	߆uЮ���iYn|��,�\�����/n2����	��,���w۾�3x��A���A4���|l=����.�N�����';�*�����fقΔe�>���_��x�	
�a��(ȁ�C1r�����m���R����;�"����I|�s�{��S2��{Dϴ��s�Ϛ�h���JKM��eَ.]Sҙ�J�!��  }`8O�g#�_�����F�E�}w�<vBw]�tO��[�N�#�7ls���1��!��;��y�d�Z��+�ߝz���o���>�1�Kؒ)m�F�T�x2���1�|�c	(�KB�ȹx����SI�v��G��K�?���t�m-u�N�L��"�뿨�O���
��d���lʭsPV2�)��Դy|�b�|�N[Z�.��z;X���Ɉyk�L����߉�q�$�5K�P���-��+N�d��C�x�3R�@�ԅVc0�R����M*���_� ���%8_Z�Z7R+K��#�$9�ǚ'���f�{�kl ��kr�"\J�{�S+l�dF$S�Ueׅ�?�P#�3��FbVLp����8ѿ��ѯ� �I����o>u+=�,�֯��*@����w^o���^�-x���p����4af`���j){�R�Ŗn�t"v�c�yu(n�R�a�b�.R./���_`=)��Y^9�� �[X_�q�D~-�(��3�RZdI��54 C#i���/���4�$�q��,@�
v����	�}�
��X��є8[�h5�������Q�*&����v�!�Ӱ��s��H��)zDJwM��I�:|)��{}��X(�~L�JE��$��u��<�Αe�l7+�Ys����nm��l}�����2�򤉴����º]���ov�i�@��n��N�S�ƞ��p�<|�xB�ãi#�Ⱦ�D� y�\�c0��.��,lXN�@����O@�2 �E��&JS)��b��|6k˶O�#|n���{��m*y�H����P~)�m��b�J��z������د�\7E_
��xrl~PӲ)H6�t�6�&L���B�����vV6K�hyPH`��\�ݲ���[�Z�>)�5��h)�n�m[P.tV����F���tk{0���j�*%�������Jck�d`�L:e7b�C鍋���e"��Ȏ��@~�-���!�����O���54�GQ"�I/	i�}�,�	����{��߻�b�p�
f�����~�'Dk�S]=p��߁��9��`���$&�����!7D:�!nr'a�.�{C�sI���}���&^�2YM�Qb�<u"m�M��1m�h�\��e�M�ڥ4�������wu'�o#���w��t�e&_�U<�j�T����WLu3��C��yO�dGX,��?��F�j�
Z���S%��?VI08nUiְ�OR���	��:s��܂�C����Lt,���x��I��$BrF�!5i�=o��&k�GRX,�DI���R��� H���
�[��CV��oc���3�F>���N���V������15����*M�tr8 r�Ӟ�H��Y�hѢ���ӅI���oV-�Fe��`�c�l��K�_)��t<B�孍}�/��	+t�^�[�7Н=�gĂ�ش-m��0rZ�=�WuSM��ۛ�C4���Ħ.X&��n9F�� ��O��I˷���(H%�'������oi�e�$0ࡗ�o�z˹��e�+}l�J��?q~o�����k�N��_�ʖ�?K�9��m��:ҙ[�i��1 }�Z,���M����t�=�� ;�	����^�h�a�atЗ^WiU\n�=r8涔��A�=+;�L�P�x5�Ӻc�S*��i�/�WQx�J�׿f|�K~P��!�)��%�h?b�_5v2�O�a�?�C      �   
  x�]R�n�0<�~�~��I<������*��腶X��@�������>�������&�1DG�)�ޅa�N>��D�
�p
;�J���.���|�RF Մ6
�*,_c7�9�}��$K�`yr}�\H�����b�.�e��?&5���qU�x���=P��$(!�<�����X4._J�h}~`J�d`�bW4y
�g"�ߘ�z��n@
�T�m��>���E�a��ªɂ�]�r�R����%�#?M��jVC�*%܅������y�C�W�a�n��EɎ;׻�m/FI�a�H+�%܇7X����)F
}�ת�]��RC37�JH*K��0�=,]<Lns:�<^��f�՟[[.4l� >��6�N�[���G���s�⊪�=!q $��[}�N�R������x�	7j��g2�F-�.�LU�a�<��c\n2����=�I��4�Ts6KX1��8f���Üb]Ãÿ,��䢻���1n�16�b՝�S�Bd=���D�	`b�$      �      x�]]Y��8��~��Lh_�,s�s��#�#�+��,Q\@�������Z������)�W�_9����Y���믮���j��������s�f;�Z�j�k�#�_�>���6���#���_���������������L���{JÈ����?�gs���9�9?���Z���V�����_��>v���4ga�է��f���x�l������9�V�`T�k���'^�.i�n�ƣ{�%�����#l���ڶg[�Vb&��T�5-ۆ�_��j1��ϩ�wl��]��+b�a�m����d�y�������̟�mE�ƾ����e�?�?���	�Z�5����"���������{��O잉K��b��{Ư�~�Z�)����d�m���篘JӪ��[����o��\l3*W���2�;������l8b�as.���%��h1i�8�n��H�*ضlE�o��oS���1���y�Ǟ4Dh�8m�g���������)�)�Ųp	x�l�L`l���2\�)7���c�le*�c�Ub*}��	d��q(�C�k`���6y��}�~B���&����;�4}C%�M{�K������Mu�V�E��R],�m����tMZn�-�\���T�@\�T{�����f�ˆ�^�qRK�s��k��Έ�}q�w?���i�!���9�Lc�SYu��W����_	=������5������38Q�fb�ak.�X>[���7,��X�bzB೧��3lU���SԧıPw�kl�!}�;lL���!��!g�Q�^w��F�
[�i'�I+5�nݸ~��s�f�>�~{���BpҚ���1C�ʑH��9���5;PmS�bq)hv���RS����) s��:��5�1��R��=��;��Ѳa�5¦�bf{��'uE�.9�x; Ç�8�B���&���><6�a+�m\?���SFX7{A�9Hi]���m.�2=`�<��gҥ����V�é,'{�q-��f�K���w&vn�|�n��e+�M�����_\�5\{Be\Z�t��lk��v��덓a���ZT��lnJ�`����o��b�Jr��a#��v����T�ݖ(5(j?^�&
��I�ꓧ,��cgk�Oy�ɉ�l����7ȴ���
���1A;6���=c�b�R�E0Nf����EC�y=f ;��B�㜆����xѴ2Fѷ?SZxV�JXZ�XJ�Y�f:la��v{װ��U_7��O1I3��gq�G� �*'�I��1�<�����8J±�1?d@C�V�Sf��U���i���3���pB6}	�I�������ʙ��&wa[��!�URy����[�ḹ�#��1��M"W���V�s)!s���m�U��@z|�-<+��4���0`��C�a4m�/�ɑ4�3�84h~�{��6��BO���^r��m9��䋮K�4���-ᱺ�v�V�RnTO��i���c��4�
�xE�@1,�i�c{���mOů}�x�Y:X�CK��??h��nKەW��q�{͵ۦ'��5(��Cܗ11��4�^�;T��r��mi���	�	W�8�\�KQ�jw}K���ѫ\����G�4�v\�eL��
���q<p����o���j����Cif�*������=�U�$�Q��8�)�_8c5���A�g]Z�1��p#ܪ�D���a��N�)�1�`��/n��{�1�~�9��)�a�	� �ܬn��ZRˈ��1>�1Ƥj���U��|\Eo��7��0ddRp��V��e�r���r�m窯�Y
��iB�|ës�t��F���
�Y�!����Ψ���	u4b����!�A�>9�6i�ʪ�_�Y��Ly��G�-0�~�T�	�	����pWM6���2Z�\��=f��i�鳾'+�u���>�wŔC�ڦcywV$��g<�����/��J��M�ml�h�L�ƪ>d��&�:�тG:��	�����od3gN�V��
_?"��ه�Ι�P���q8բ�sq,�,�	�1���,�F��zO%0��`Vg���#s�h�!�p�\���x�vxjb����m�:�JGtaڠ��r=Q�2#�E9t'/��zL��bǿH�՞����Q�
����ԩ���#!��%�����>u����s��	���P�k��`�]y*��E5��x��n�����W�o2=Q-{�X;O�a��;�cϰ�<
�����m���3��|*d�>[5у�R�D,��x���0Ҹ���=W��A��˱�z�wV���.�1@c�/��\I��|����R>����G�CJ�=��Tc�_���Y'~�#��<b3�?$�����4�/�յPa��{�څ���sC$Fʌ���,���ۣ*��*���^uվ
�o�㌔�}�����M�mN�j�|6鰣�|Ax���APg�9ڕ8d��P+^5�F;�H�pe����:�:Ҕ@��!��,x���a �;T�� �����zK���S�?z�Dko>.��]��E��[�b�ֵ���u�'���C��c�o���PKP#��ۙ���$Ԓ�p-��2ħf֋�i�/���Q��\�ٿ�s��5��48�ޠo�3��6�ꈖ���F�J�"��7m椆���t�m:]�rӤE�����e)t���K����Ѫ���q�rJ�}�����Kd��3D��!^���y6�E�m�w����e���U��0�1g� .��L􈀵%՜��l�X��U�q6�t���_��~����?\�Q�k�!G�\��E4y�
ڕO�oF�<mKAϤ�	��ˁ�w,?C�F��1ͽ��B���Grj
C\Y@~Zl(�Ŀʾhq�+tQh�CMS�v�0�%��d�����)��s������Y@�3ReNl��h�śP-�\�"����8�FN!���|���q�T��N�S��M7ܰ��3�t����h<tȩ�{�T;������$X7��4i%�塈9p\���\�U�l;u�k�t?f��0g
q,�wJ���>fS(�M��(�,��S>Ԃ0�&b�P+SƼَ#q5��Zd�+�0�E|j��G� d�����_���T�g2��Ss�(��@���8�QS6�
�nȴ��u�4u�M=CH�}�^T�
!��|�+��ѓ�����fE�GȳŒ��8f|��5�Xn�d���@��hg�46E�=,��,�0�̇^7�P;����%hDGb�Y�թd^���ޠ	&�#�)�DV|�Y�r)�5���V~�@j=���Z��S^��p��FG�C58r��]f�`/���������c���yI�����������D)E]���%{��Q(xUH�:�,�zk��!�HA�¼g9��e;�|�>L(��z��rs�G���8���[�0 U�WJrA���5�C .�ub=r����l�x*u9ʟ2�w3�k�˦�p��K�aq�1S6+@�M4p�"-L8�H����{��z�Q�������K�
A�K�)���Zӑ�*��Y�� ��J �HgU�PT�c,�c�����w���^��x�.�נ*��*�B@.���BNA������+�M�S�b�>)�u����9o��Hx �@݆WI���R�ؼ�J]�����7��)�����x��oX����r;b��C���\����3�b�&��s�QO�9�>���xG���c��7�=rf�\�Y�������x��]��
���2AT��|��B/��N��:W47�����	0�rQ$^ �|Ws`r���2�$^�`�R8ug�B�l*^3,r�����$��ڊ��'��h���b]x`���Da\�S�ez:K�rN���4E8�����T.�]����\�\�gG&�2��s6'6��2S��En�����۱��A����YM�{$��=���|���:���Y�n���l͜���� ��	�*<�__gD}�M�������>緀��@��P:�c:    �1����f�SSm&d�*��W͵�6�J"��m��k�F^�ep��?x|,��8����к���-1�	���� ���4%�~Ua+�\���3s�c���y(��מa")�ad���,�H&c*m�ʕ�S:J�<SN�\uF��ƹ)��^���.;���*Bi>^��rs�Uy�<���X�c:d2C}�p�B��()���a������y�W+h�ؤ;\��}#��u$�����T�>���|%��#c�J�f���N	$���������'�uQ�� �餤���Z����d�9�5c
�3��1(.�|�O:��ݪ�.i��r�18E���P�S�K��V��� �S�����5�#ZX�Z�1��O�I�J��)�U�4��y@|�1Xp����_��eE��n�;E�:i�͒��ᳩD��������w��|4�]6ԃ|��`���ם��OzY�t�F�qf'a�k�Ζ�I��'�9 #� ��X%d������!�x\p�Ğ-��b!���Wf���b~�.�Z�{5wX@xs�V�E��c,A��D�g~��b��:��y�&ǭc��j�z���_���G�g�3�т$w$L��	)����G&�3s>�N�=�0;�#��{HF���2��� �F���w���[ۉ�� @���!�x��,m+��@o_RF�C��	��He��@(�h՘���\U$�76�'w>��
�����kO�D��<	_���Μ�GI=���wm�L�������1P�'�����;�j8�H���L����t,��ܘ�b��=	�)s�C��ϝɌIV��2�}�[�Ǔ��<B����8��1�a�@F��<�&�FG��	�F<K��@:���D�*�,f_���A�@@�Vf2l�!��I�G�Cr^�廴 4ŕ�/|�0�p%b歬��Z��Y�']�Xhca�y\>����x�	�D9hL�FĂ� X;����WQa�H����aT
���LlbsO�TЮ>isLP5�wP�K���gG���)��i�0� m7�(Ҭ��gU������<w���K�rN��ы^��p��Cx������F�6a���J���CO��@�W%�����Zr�|��� ib�w$��L�S�O�W�R���0�J�=����4���pHXe�� �����@�����B��A�ߞ�٧���%���9Ҥ���n���p9+�������;0̾��U�9.�Z�'H�������G����'�-h�B:vȡ��v��@�:h�0[M�^�y��#����r}�Lp��U�$C������b: ���E-�d9��j	`�����|�#e>e���/�yS	N��
z�R���~
�a�H������� c��|P�AK�уhz:��Ul�����&Gӌ�gi��Fp�],�q��R�����4P��@���Iq�1�EF��<�HMU�J�I��Q��Q�$Do��3�l>��^�@ |��ۛ���jp���cP��)��0?4�e�S�N|�AC�XJ�YN�s�)"T��2��/H���xӺM[~�C?
)Q��,0�1a��bz���R�H�Hq`��Ȇ�Ҧ���ù���ιk��� �=�����;2�?A�'��2 �����S��fc�\䤰���[5vư���cq��W�^Բ�����L�A]�avo/Ꜽw�HTM�nS�^P�38�����>�_u�A��;�:n����3Y}TW,O��7_z-̷�ɫ��M��1��;��7K��i)�zm�`�1�ee$���{^�SA
���i�B�� �y�u��YJ�j�ZG�r�G�[��qPUrŢo�۫�D'z�IΆ���Y�3h�0��]"ĝ-Y�0�.H�'RI��颉0l���:ՃFKpKfކ,d��r��*��>�'�ʽ�u�wx�-�.�ȅe�sB-�hbJ�E��ƛ�>K����{�ҕ�'�#ZC(�G\����.Đܽ+�(gB8L碮�F����7�ء/j
A�|P��qD���Xh*5��\5H�B�;&�Z x�3M���,���m^N1��&�XL�FJ��&-1^U�Up��-�F��rY��<�Bʫ0�^Ta:��d�������D�\e"z����ubɌᖯy��JM�=H"�Nx��O�Z�_����O"�-=���y4E%?�]ʒ�TY>�	�gw�O��ĈM�g�����]�=Т^�:��`�|L#-V� � ���f �S�����'Á �b�Dvd�+�%���4�\*CIu|��g���	aU�6kE�5��o�� },(9f<���H!���w�=6�8Q�}�\���(���{]P��WG�P��	_��p����2�3���{�����#ˇt��E���=���\���ñ�.�|L]�L��#L�n�7.xM�G�0�i>��P��-F�2<B���F �����$,ЄV�
�"���٬q���[�S ��s?;$I֪cE�%�h��D�d���a���N�$���k����?�o�Qk�l��0`�J��֊\�v��q"�H�#�a��jN������Dt�L�q�B>q�
	F��)��~�˕\x�~�*i39�������:����_�W���޴�� :w}&��Ja���@l�$}��'<���
Gޝ�z�"��@�l���.��T����d�GGl�qnm%�@���l�b9�p�r��UQ���;�Ǣ�B&�y^s�6���_}Zj���x��ZO��hw�����Pn{�D��u�XJU�!�p�&��te��6��!_�o�0��/��Q��f�>�0���%��B V��%:��$<L�璢s�t"���@���ᢤ�!G�,��,����]�K�D��֋����,^�Oܞ���U�]*�ꔐ*��"����u�Â����#����\�M�#i����Α�x�d��zU��ӎ�HG���Y�c��s2_o8�]9�9�]^��X(�u�3f|�
!��P�A#���ì�M��c�1*
b����^��I7�=��d��W����O�1�	-��	���!��=u�٣f"�\��C56Q~"b+x�!������ڷ�D�1_�� �W���nU��\�,���8��F�y��w��RP 6��S�o���M��6�t'A����bf���B)mY�:Բ ��"���������u���R�� 5|�"SB5�J����e��>E�K�x���� "��)ָ��
ra��LueB��ZH3
{���D��Ł�Q56EOi���d�I�\�jgm���Ai�%��ff�y"���"�I�߃*�lؐ�2��z���yv�@�ώ�h��x�n�s��#���p�P����Ec������)�"&9��G]��Sp>}��1خ�M�Q�+E����o��Y*�J&�T<0�b8��Y���N|�@�1`>Ik�r�X�1�d��L����A��D���Ρ���Y���m{�x3��'��L�	RkM�ŀ���H���z=(�:͡�s��Qu�V��2)G���b($g�����z�0�y��F�N���)�$\o���5;���Tz�	��I��˂U�}P��0���i�k�k}��໳�q�T�8�Jd�:���Y������0��r�������D�o��`�.﯅��+�� �C>���U�9�*m�U�1�xU�8���]� <�GR����;���q�<O��f��E�of�<�*
�2Q��� ƊCv��Ń$����@��[���P :w�=��p�8�������h�	�����
�aӥ�e�mLep 4���7�v(�2t�ů�&�����0�F4��B>�`���J�8�\͍Vs�g������y>d���m�V�v����|��T�j�
���k�2j 
5����ގ
�3)�'��C�2y9��%/{�2z�����O�C/e����6    �$5ei�X���t?�&���l�E� eO�=���*�Չe���ُ�|�~��+���8(��&�א� x!��˨|X׏ �G���G]L{Q���ejnd��ˠyW���:Rm�ٲC-}�=��=	B-�0���n4t+%�cS����=�p�Oy�p�2{>�r�-\�.~Uˈ��&��}�ĜK��|������bΖ�s>,v��`���E��D�V�w/�[�����8t,�H���%п�	x��69�E��	P߆U"���{��l�#c�w`\y��xG�
Q7(�Q�1�ЍF���I0Nl+Yx�ۻ{���u+������Z�tV��z˗�TU~M���qoM��\��,X�F@�f���s]� ��:�")�]������?�h�p|�L&����/���-�(ڍ�����LF�/�|�>���H���;|M&v�L$`��_ c�:s�53]Z�+�� ����W�bW���U�m��3�ǔ�`�/��)�tM��"`�޶�����`cAu�{����0e�Jd.��C^�-�øظ���m� ��Qx���GExv�]�:�9�a�9[�i��c�� P?N�����{/�^��:>m�(̋�FN�J�dA�������ⳮ�E����Jx�#w*�|�`KBI�9z(G�N|Z��xMR��_�;��!j��Mr���A#*�*��86=5cxC$!�B�<�Υkҧ��dC�:�iG^������k�%�Q���j.�ƦS����gv}K��Cc3;�F�\hJ�:QM�	($�DU].��ן㩩,OW��G,�jk����4e���ve$�4NYU%]x
"�����5V]�ۣ�|.�6ʍ,;+COEft�x4K�BCc�N��$��A�R������̥~�����B[vA�2 oH}̘�H�6�ET1D���h�$������L�L�����b�\�Ŏ~�·��2#�Y�D�̟���ȧ�LCE��N�0*1�":tJu��fW����蜳݋s�yQ�F��`.`�8?Y�<u$йT,oqU�"rrl�B�i���.�& �J+������?n�B7�N�I��JEm2�Zb�~9\A��rS��[\eq�,��~�G_��O�h�n�����EnQ���e�0Im72��|��U@vCܳ���e�bc�(��o�����A<�3��{��D�+|E�:ȯMg{�l�5��(�'*�m����o�E":R)�E���=�����$q���fVW�'[��&#�uH���ވB�YE66eX]��&����)��.�eB�}Jݽ�yutIF��K�V�[ڔ�_t���"�
���Z'�����Z_�C63H�A�1���k8�ֈ�{vz��A�C���j��a[�u�ꁺy{��]���{<�T.�l����:�Y3���� Pݷ&����~��j��{UF�59���g�#���e��"�����q1��p �ɛ�4���O-��0ؚ��uIS����?�nn������&�(m��&R;_��� ����,��f��/-�;��X��I���tԪO�?��=�]]�� �%3H�"B��TrB�+�X�_֧pA='Ѩ�M�����|Bf�y�͈�c�S�Sr�c'����d���s���L^W6�Z���q�m�S���*⭢c:[(y3�//S��A�	�H#�kb~y���
*�j�����З�'��j�(����I���+��}�P�XA�@G^D]����u�~ @>3�6	���Ƀn��� ���GVI��b��	�a� u�����͛�3��	1Ȑ�D30R�飤nF@,���-���h���c*�s�N�a(���u�`���VU�=�����]L�H|ry���Y����>��a�^�:�
żi���|�Gs�)�RKD����C��z=y�~T\oE߳Z�U�k{�Sm	�1�So����g�JFF�K���+��D�2@�0Z]��d�@*o%e6Zr�e�	+{1g5�f���p|�r���6H�]��gaD�Ot�f���v�-�^�SF`�y��:�d̈���4I��g����3�J����N~�a�6�B2�1��꺔�U5�<��%п]0H�>��Dl���o���ZN�ѕ���$�e���h����Q�MZpw�r}�|����������eKk�
p�G/ј+�#M�&����׃K5_��ڴ٘��*����$G��������ԠoP��{�1V�w�诟�f�F$z}����2_O�i�}Lc�Pp����lz�%�1&ʕ���̒%�jg�@������u*���� ���B�e2�Ao��__��W�������z�J�U��y��eO�[�T��'�v ފ���օ=PC��� ]��߾(����'�z�\�=���Mi3UW��/A%~�6E�ͫ4�P�8����	���2^{QO"�7`u�w��o�l̾�-��I;�rUK�|3����XKN��==��a� �{����*y+���ޜ��[Zs>!(��B�J � C=�<�������޹"��2��t�V��p �wY2Y���EuaC[7��%+�tQH̼���e�'� �Ҳ��e՚�t�b���$3;-��bƗ_���^�%��WS���j7�i}�\�gnޖɪ!Op��+7B:v���y-��TG��b�-��VU��C8�f����!^
��J}6�td�����Ȩ��>�A����f�*L���ƱUf)�M51�z��r������
���+���$�-��Ȥ��m�k��9�����h��������S� ���҉!֖rϦF��x���N�f�m�i�M�`�wq�J�5U�! GM.lȢd Wо��<z���Q9�t��ސ(����k�y���v��.��
}<�Z���&z���������+.)Bj�rH��a���I]�p�MޯS35�����r��q^��`돏��8��C�l�>�P�J�	�q��U�1p_�uxw��m�sx��Z�5.N�@w�ʽwarI�n'"	P���^�J��V�Q������e��b���Lٴ���x!�k��G�^���h��w��LT�n>=�И�t�3�]�!� "��T�6��0�קy���P���G����� -�d���rӵ�z�#��=�ԛ$���C-�ߌ��{�EUqZ��]Gy{_������;:;��m���i~AK X%��\x��^�U_�=ڌ�l$T�D�X����F6f{e^^�n�t,^6��}FNԫ�]xp�l�T�tK�a�]����i�.7s-LP��"��)��u��kCbѣ����PEC������va������*�l)�\#�;E��m�$��x���crqlx���%4{u���q��[��
�I!��fxhM����s	�sC����4��Gix�B0ȼ&!��J?�2��k�xU/� ��O��*Ko�2�����������킷����|���Q�][I�I�ۼ��k���yQ��3Sl��
[t��+U�Jt!A��NfW�?QŰ;�!-
�3u�2EM�o��;K`G	M���s��\t�*�O�R�y�y�����9��D|��d/e��+H��5�ֺM?���UR�B$�eU"��&B�ݲ�IF-�L6K@g�@\�lH�0�@��[�]�ǖ	 ����h.�_�|��?��Nk�G�\? �4��]+Y�`1(�+��=,�	�,/�:�.=1� (�rxzIP� v�;��p=$�{�8�gf"ď��Η����CWۮnsB�T�bCpy^g����o_������Њ�{%�8�@�>?�FW��"�,WD}4�s:j�֯�s-�����ĔE���_P�h��ɰ&�^QJ�aly�����1fx����o4���ƍ�;����h2�qUl���'�l���A���zlt�#��Rv�_�߄_ \�ͱ�Z�$D��%�|��U���b��zPX
",�(��E���|L��[�����9����G��,�=���@a�W7<��6�<w�XJD?�F�d{�y=�/'u����S�SO>z�WS ���   ��f��T��H��� ~P9C��[p�ok�����d���t��Y������okoz���'����P�/�E@|�y�3����B���QˊF�����4�g�q�\�>�4a|�'ߋ��&��=�S*��`_5]"+��:�>�r��;rMH�qk�;�Yy5���y9�0�� ʻd�'E %>�a/��71�o�Qx.t?��>#מ�y�e����o��¼�ǲ��u�j;����'��֍�����\�}���$�ٓP:���{�Uk�T���|�"��f�������E7���/X�����$��hf�߂��������o�RM$Ϲt�b���wKx)�	s!����,7�Ţ�wLB�i�Vtb�Je%�"��;�{`k=��xߵٰ�>ܡ,d%S��6���q�2��nا]c.��PO޴�HQ����K�?�W��>7>^^���*����/r�8����םN�����rJ�z]w'>�V0:y=�_�[E)��1*lo����&2�e���R^N�x�__��]=n��*�r�)�G�=|��DƄ/��V�}=UI-�\Y��I�4)J'��~Roׄ�v:_ͶNj�t��l�2�#n��!�/�����wMݮ�vY[/��F�y�P#���h4��S�����l�2�����=���m��~Ȟ'��,���!"�犲��I�_��Y�������FwxsO�@φ�[X����&����jn���w��U���wm�Mm�LTVq����t�8Vh��u�E��	����йh�.Gpt�Z���P�x*�׺`��loW2��PD���EAވn%~j�L[�W]{���Cm��2Y���//�z�7Q#�;ic��w(��7z�sA�<~S�7��"�3��Q�E�LI�G�i�QH�-"�qv��P'�k{�;x�^:W�s�҉��W��Q����x�&l��,w.Y(�B��F	{B�����8�V�stm����5��9s���f����M�-c��NXzr��q���a��p�l�55]iҖ���vYkb�2�1���m�n���|�]�����;z@��D츓���G�����&�l�:��=ǯë����A��G�dR?)���P��}n-���!��Ț9S.o��"���7ɲ�,��&�YU~zqg:����(k�U�s��C�V�;�}c��1u����dg�y>��/d��$��'ko�#תf��M���i[	��<��r��! Y7�W/.�j���ޮ'e�Cw?^�!���Wkd�dMq��:��a?qE<�T4Qu󜅑��3�L_�,��
D%�/����	���Tr�#(j&��H�M��r���g�u5��8�@���r(�:gƘ�`��w2����]^��F����KU/ëK�}<�B�B(B� @�k�u�d�{���u��Ro�l�l�*���`=1/�rVz�'S˄K���Z,�E��SŹ�M4#/�b������)#����^Nr��$sq��S��|K�n4�y��%yI�J�8���ηن��������xHߊ     