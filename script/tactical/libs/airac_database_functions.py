from decimal import *
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def truncate(f, n):
  return round((math.floor(f * 10 ** n) / 10 ** n),n-1)

def format_date(x):
  d,m,y=x.split('/')
  return y+"-"+m+"-"+d

def create_ddr_databases(conn,db_name="ddr_databases"):
    
    conn.execute("SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0")
    conn.execute("SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0")
    conn.execute("SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES'")

    conn.execute("DROP SCHEMA IF EXISTS "+db_name+"")

    #Schema ddrdb
    conn.execute("CREATE SCHEMA IF NOT EXISTS "+db_name+" DEFAULT CHARACTER SET latin1")
    conn.execute("USE "+db_name+"")

    #Table airac
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`ddr_databases`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`ddr_databases` ( \
      `airac` INT(11) NOT NULL, \
      `ddr_name` VARCHAR(25) NOT NULL, \
      `ddr_2_days` INT(3) NOT NULL DEFAULT 0, \
      `ddr_3_days` INT(3) NOT NULL DEFAULT 0, \
      `ddr_4_days` INT(3) NOT NULL DEFAULT 0, \
      `ddr_individual_days` INT(3) NOT NULL DEFAULT 0, \
      PRIMARY KEY (`airac`,`ddr_name`)) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")

    conn.execute("SET SQL_MODE=@OLD_SQL_MODE")
    conn.execute("SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS")
    conn.execute("SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS")
    
def create_ddr_database(conn,db_name):

    conn.execute("SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0")
    conn.execute("SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0")
    conn.execute("SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES'")

    conn.execute("DROP SCHEMA IF EXISTS "+db_name+"")

    #Schema ddrdb
    conn.execute("CREATE SCHEMA IF NOT EXISTS "+db_name+" DEFAULT CHARACTER SET latin1")
    conn.execute("USE "+db_name+"")


    #Table airac
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airac`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airac` ( \
      `id` INT(11) NOT NULL, \
      `date_start` DATE NULL DEFAULT NULL, \
      `date_end` DATE NULL DEFAULT NULL, \
      PRIMARY KEY (`id`)) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")

    #Table airblock
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airblock`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airblock` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `name` VARCHAR(25) NOT NULL, \
      `boundary` POLYGON NOT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `name` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 80485 \
    DEFAULT CHARACTER SET = latin1 \
    COMMENT = 'INDEX(name) ?'")


    #Table airport
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airport`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airport` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `icao_id` VARCHAR(10) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`icao_id` ASC), \
      INDEX `icao_id_indx` (`icao_id` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 109154 \
    DEFAULT CHARACTER SET = latin1")

    #Table airports_ecac
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airports_ecac`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airports_ecac` ( \
      `icao` VARCHAR(4) NOT NULL, \
      PRIMARY KEY (`icao`)) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")

    #Table airportset
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airportset`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airportset` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `sid` VARCHAR(10) NOT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 22543 \
    DEFAULT CHARACTER SET = latin1")


    #Table airspace
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airspace` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(25) NOT NULL, \
      `type` VARCHAR(5) NOT NULL, \
      `name` VARCHAR(50) NULL DEFAULT '_', \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 60066 \
    DEFAULT CHARACTER SET = latin1 \
    COMMENT = 'INDEX(airspaceId)\nINDEX(name)'")

    #Table airspace_has_airspace
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airspace_has_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airspace_has_airspace` ( \
      `parent_airspace_id` INT(11) NOT NULL, \
      `subairspace_id` INT(11) NOT NULL, \
      PRIMARY KEY (`parent_airspace_id`, `subairspace_id`), \
      INDEX `fk_AirspaceHasAirspace_Airspace2` (`subairspace_id` ASC), \
      CONSTRAINT `fk_AirspaceHasAirspace_Airspace1` \
        FOREIGN KEY (`parent_airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_AirspaceHasAirspace_Airspace2` \
        FOREIGN KEY (`subairspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")

    #Table sector
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`sector` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(25) NOT NULL, \
      `name` VARCHAR(50) NOT NULL, \
      `type` VARCHAR(5) NOT NULL, \
      `category` VARCHAR(5) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 64186 \
    DEFAULT CHARACTER SET = latin1")


    #Table airspace_has_sector
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airspace_has_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airspace_has_sector` ( \
      `airspace_id` INT(11) NOT NULL, \
      `sector_id` INT(11) NOT NULL, \
      PRIMARY KEY (`airspace_id`, `sector_id`), \
      INDEX `sector_airspace_link` (`sector_id` ASC), \
      CONSTRAINT `airspace_sector_link` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `sector_airspace_link` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table configuration
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`configuration`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`configuration` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(20) NOT NULL, \
      `acc_id` INT(11) NOT NULL, \
      `num_controllers` INT(11), \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC, `acc_id` ASC), \
      INDEX `fk_Configuration_Airspace1` (`acc_id` ASC), \
      CONSTRAINT `fk_Configuration_Airspace1` \
        FOREIGN KEY (`acc_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 58161 \
    DEFAULT CHARACTER SET = latin1 \
    COMMENT = 'INDEX(ACCName, name)'")


    #Tableconfiguration_has_sector
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`configuration_has_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`configuration_has_sector` ( \
      `sector_id` INT(11) NOT NULL, \
      `configuration_id` INT(11) NOT NULL, \
      PRIMARY KEY (`sector_id`, `configuration_id`), \
      INDEX `fk_ConfigurationHasSector_Configuration1` (`configuration_id` ASC), \
      CONSTRAINT `fk_ConfigurationHasSector_Configuration1` \
        FOREIGN KEY (`configuration_id`) \
        REFERENCES "+db_name+".`configuration` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_ConfigurationHasSector_Sector1` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Tablecontrollers_availability
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`controllers_availability`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`controllers_availability` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `airspace_id` INT(11) NULL DEFAULT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `max_controllers_pos` INT(11) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `start` ASC, `end` ASC, `airspace_id` ASC), \
      INDEX `fk_ControllersAvailability_Airspace1` (`airspace_id` ASC), \
      CONSTRAINT `fk_ControllersAvailability_Airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 493697 \
    DEFAULT CHARACTER SET = latin1")


    #Table flight
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flight`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flight` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` VARCHAR(45) NOT NULL, \
      `ifps_id` VARCHAR(45) NULL DEFAULT NULL, \
      `ac_id` VARCHAR(15) NOT NULL, \
      `tact_id` INT(11) NULL DEFAULT NULL, \
      `ac_type` VARCHAR(8) NULL DEFAULT NULL, \
      `ac_id_iata` VARCHAR(10) NULL DEFAULT NULL, \
      `airport_arrival` INT(11) NOT NULL, \
      `airport_departure` INT(11) NOT NULL, \
      `operator` VARCHAR(3) NULL DEFAULT NULL, \
      `aobt` DATETIME NULL DEFAULT NULL, \
      `iobt` DATETIME NULL DEFAULT NULL, \
      `cobt` DATETIME NULL DEFAULT NULL, \
      `eobt` DATETIME NULL DEFAULT NULL, \
      `lobt` DATETIME NULL DEFAULT NULL, \
      `original_flight_data_quality` VARCHAR(3) NULL DEFAULT NULL, \
      `flight_data_quality` VARCHAR(3) NULL DEFAULT NULL, \
      `source` VARCHAR(3) NULL DEFAULT NULL, \
      `late_filer` TINYINT(1) NULL DEFAULT NULL, \
      `late_updater` TINYINT(1) NULL DEFAULT NULL, \
      `north_atlantic_flight_status` TINYINT(1) NULL DEFAULT NULL, \
      `flight_state` VARCHAR(2) NULL DEFAULT NULL, \
      `prev_to_activation_flight_state` VARCHAR(2) NULL DEFAULT NULL, \
      `sensitive_flight` TINYINT(1) NULL DEFAULT NULL, \
      `operating_aircraft_operator_icao_id` VARCHAR(3) NULL DEFAULT NULL, \
      `runway_visual_range` INT(11) NULL DEFAULT NULL, \
      `arc_addr_source` VARCHAR(1) NULL DEFAULT NULL, \
      `arc_addr` VARCHAR(6) NULL DEFAULT NULL, \
      `ifps_registration_mark` VARCHAR(7) NULL DEFAULT NULL, \
      `flight_type_icao` VARCHAR(2) NULL DEFAULT NULL, \
      `aircraft_equipment` VARCHAR(250) NULL DEFAULT NULL, \
      `no_cpgcpf_reason` VARCHAR(1) NULL DEFAULT NULL, \
      `ddr_version` INT(11) NULL DEFAULT NULL, \
      `ddr_source` VARCHAR(80) NULL, \
      `individual_day` TINYINT(1) NOT NULL DEFAULT 0, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `ifps_id` ASC), \
      INDEX `fk_Flight_Airport10` (`airport_departure` ASC), \
      INDEX `fk_Flight_Airport20` (`airport_arrival` ASC), \
      INDEX `fk_flight_ifps_id1` (`ifps_id` ASC), \
      INDEX `fk_flights_airac_1` (`airac` ASC), \
      INDEX `fk_flights_version_1` (`ddr_version` ASC), \
      INDEX `fk_flights_source_1` (`ddr_source` ASC), \
      INDEX `fk_individual_day_1` (`individual_day` ASC), \
      CONSTRAINT `fk_Flight_Airport10` \
        FOREIGN KEY (`airport_departure`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Flight_Airport20` \
        FOREIGN KEY (`airport_arrival`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 39346 \
    DEFAULT CHARACTER SET = latin1")


    #Table trafficvolume
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(20) NOT NULL, \
      `name` VARCHAR(500) NULL DEFAULT '_', \
      `category` VARCHAR(2) NOT NULL DEFAULT '_', \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 112667 \
    DEFAULT CHARACTER SET = latin1")


    #Table regulation
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`regulation`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`regulation` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(20) NOT NULL, \
      `trafficvolume_id` INT(11) NULL DEFAULT NULL, \
      `airspace_id` INT(11) NULL DEFAULT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `slotwindow_width` INT(11) NULL DEFAULT NULL, \
      `slotslice_width` INT(11) NULL DEFAULT NULL, \
      `reason` CHAR(1) NOT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC), \
      INDEX `fk_Regulation_TrafficVolume1` (`trafficvolume_id` ASC), \
      INDEX `fk_Regulation_Airspace1` (`airspace_id` ASC), \
      INDEX `reason_index` (`reason` ASC), \
      CONSTRAINT `fk_Regulation_Airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Regulation_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 28545 \
    DEFAULT CHARACTER SET = latin1")



     #Table  flight_atfm_info

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flight_atfm_info`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flight_atfm_info` ( \
      `flight_id` INT(11) NOT NULL, \
      `excemption_reason_type` VARCHAR(10) NULL DEFAULT NULL, \
      `excemption_reason_distance` VARCHAR(10) NULL DEFAULT NULL, \
      `suspension_status` VARCHAR(2) NULL DEFAULT NULL, \
      `sam_ctot` DATETIME NULL DEFAULT NULL, \
      `sam_sent` TINYINT(1) NULL DEFAULT NULL, \
      `sip_ctot` DATETIME NULL DEFAULT NULL, \
      `sip_sent` TINYINT(1) NULL DEFAULT NULL, \
      `slot_forced` TINYINT(1) NULL DEFAULT NULL, \
      `most_penalising_reg` INT(11) NULL DEFAULT NULL, \
      `regulations_affected_by_nr_of_instances` INT(11) NULL DEFAULT NULL, \
      `reg_excluded_from_nr_of_instances` INT(11) NULL DEFAULT NULL, \
      `last_received_atfm_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `last_received_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `last_sent_atfm_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `manual_exemption_reason` VARCHAR(1) NULL DEFAULT NULL, \
      `ready_for_improvement` TINYINT(1) NULL DEFAULT NULL, \
      `ready_to_depart` TINYINT(1) NULL DEFAULT NULL, \
      `revised_taxi_time` INT(11) NULL DEFAULT NULL, \
      `tis` INT(11) NULL DEFAULT NULL, \
      `trs` INT(11) NULL DEFAULT NULL, \
      `to_be_sent_slot_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `to_be_sent_proposal_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `last_sent_slot_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `last_sent_proposal_message_title` VARCHAR(3) NULL DEFAULT NULL, \
      `last_sent_proposal_message` DATETIME NULL DEFAULT NULL, \
      `last_sent_slot_message` DATETIME NULL DEFAULT NULL, \
      `flight_count_option` VARCHAR(1) NULL DEFAULT NULL, \
      `normal_flight_tact_id` INT(11) NULL DEFAULT NULL, \
      `proposal_flight_tact_id` INT(11) NULL DEFAULT NULL, \
      `rerouting_why` VARCHAR(1) NULL DEFAULT NULL, \
      `rerouted_flight_state` VARCHAR(1) NULL DEFAULT NULL, \
      `number_ignored_errors` INT(11) NULL DEFAULT NULL, \
      PRIMARY KEY (`flight_id`), \
      INDEX `fk_FlightATFMInfo_Regulation1` (`most_penalising_reg` ASC), \
      CONSTRAINT `fk_FlightATFMInfo_Flight1` \
        FOREIGN KEY (`flight_id`) \
        REFERENCES "+db_name+".`flight` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_FlightATFMInfo_Regulation1` \
        FOREIGN KEY (`most_penalising_reg`) \
        REFERENCES "+db_name+".`regulation` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table flight_cdm_info
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flight_cdm_info`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flight_cdm_info` ( \
      `flight_id` INT(11) NOT NULL, \
      `cdm_status` VARCHAR(1) NULL DEFAULT NULL, \
      `cdm_early_ttot` DATETIME NULL DEFAULT NULL, \
      `cdm_ao_ttot` DATETIME NULL DEFAULT NULL, \
      `cdm_atc_ttot` DATETIME NULL DEFAULT NULL, \
      `cdm_sequenced_ttot` DATETIME NULL DEFAULT NULL, \
      `cdm_taxi_time` TIME NULL DEFAULT NULL, \
      `cdm_off_block_time_discrepancy` TINYINT(1) NULL DEFAULT NULL, \
      `cdm_departure_procedure_id` VARCHAR(8) NULL DEFAULT NULL, \
      `cdm_aircraft_type_id` VARCHAR(8) NULL DEFAULT NULL, \
      `cdm_registration_mark` VARCHAR(7) NULL DEFAULT NULL, \
      `cdm_no_slot_before` DATETIME NULL DEFAULT NULL, \
      `cdm_departure_status` VARCHAR(1) NULL DEFAULT NULL, \
      PRIMARY KEY (`flight_id`), \
      CONSTRAINT `fk_CDM_Flight1` \
        FOREIGN KEY (`flight_id`) \
        REFERENCES "+db_name+".`flight` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table flight_intention_info
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flight_intention_info`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flight_intention_info` ( \
      `flight_id` INT(11) NOT NULL, \
      `intention_flight` TINYINT(1) NULL DEFAULT NULL, \
      `intention_related_route_assignment_method` VARCHAR(20) NULL DEFAULT NULL, \
      `intention_uid` VARCHAR(10) NULL DEFAULT NULL, \
      `intention_edition_date` DATETIME NULL DEFAULT NULL, \
      `intention_source` VARCHAR(10) NULL DEFAULT NULL, \
      `associated_intetions` VARCHAR(20) NULL DEFAULT NULL, \
      `enrichment_output` VARCHAR(20) NULL DEFAULT NULL, \
      PRIMARY KEY (`flight_id`), \
      CONSTRAINT `fk_Intention_Flight1` \
        FOREIGN KEY (`flight_id`) \
        REFERENCES "+db_name+".`flight` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table flow
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(25) NOT NULL, \
      `name` VARCHAR(250) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`airac` ASC, `sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 123120 \
    DEFAULT CHARACTER SET = latin1")


    #Table flow_has_airport
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow_has_airport`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow_has_airport` ( \
      `flow_id` INT(11) NOT NULL, \
      `airport_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`flow_id`, `airport_id`, `sense`), \
      INDEX `fk_Flow_has_Airport_Airport1` (`airport_id` ASC), \
      CONSTRAINT `fk_Flow_has_Airport_Airport1` \
        FOREIGN KEY (`airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Flow_has_Airport_Flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table flow_has_airportset
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow_has_airportset`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow_has_airportset` ( \
      `flow_id` INT(11) NOT NULL, \
      `airportset_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`flow_id`, `airportset_id`, `sense`), \
      INDEX `fk_Flow_has_AirportSet_AirportSet1` (`airportset_id` ASC), \
      CONSTRAINT `fk_Flow_has_AirportSet_AirportSet1` \
        FOREIGN KEY (`airportset_id`) \
        REFERENCES "+db_name+".`airportset` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Flow_has_AirportSet_Flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table flow_has_sector
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow_has_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow_has_sector` ( \
      `flow_id` INT(11) NOT NULL, \
      `sector_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`flow_id`, `sector_id`, `sense`), \
      INDEX `fk_Flow_has_Sector_Sector1` (`sector_id` ASC), \
      CONSTRAINT `fk_Flow_has_Sector_Flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Flow_has_Sector_Sector1` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table waypoint
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`waypoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`waypoint` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `sid` VARCHAR(8) NOT NULL, \
      `type` VARCHAR(2) NOT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`id` ASC, `sid` ASC), \
      INDEX `sid_indx` (`sid` ASC)) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 491467 \
    DEFAULT CHARACTER SET = latin1")


    #Table flow_has_waypoint
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow_has_waypoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow_has_waypoint` ( \
      `flow_id` INT(11) NOT NULL, \
      `waypoint_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`flow_id`, `waypoint_id`, `sense`), \
      INDEX `fk_Flow_has_WayPoint_WayPoint1` (`waypoint_id` ASC), \
      CONSTRAINT `fk_Flow_has_WayPoint_Flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_Flow_has_WayPoint_WayPoint1` \
        FOREIGN KEY (`waypoint_id`) \
        REFERENCES "+db_name+".`waypoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table opening_scheme
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`opening_scheme`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`opening_scheme` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `airac` INT(11) NOT NULL, \
      `configuration_id` INT(11) NULL DEFAULT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `info_origin` CHAR(1) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `unique_id` (`configuration_id` ASC, `start` ASC, `end` ASC), \
      CONSTRAINT `fk_OpeningScheme_Configuration1` \
        FOREIGN KEY (`configuration_id`) \
        REFERENCES "+db_name+".`configuration` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 316779 \
    DEFAULT CHARACTER SET = latin1 \
    COMMENT = 'INDEX(airpaceId)\nINDEX(airpaceUId)'")


    #Table regulation_period`
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`regulation_period`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`regulation_period` ( \
      `regulation_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      PRIMARY KEY (`regulation_id`, `start`, `end`), \
      CONSTRAINT `fk_RegulationPeriod_Regulation1` \
        FOREIGN KEY (`regulation_id`) \
        REFERENCES "+db_name+".`regulation` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table sectorslice
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`sectorslice`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`sectorslice` ( \
      `id` INT(11) NOT NULL AUTO_INCREMENT, \
      `sector_id` INT(11) NULL DEFAULT NULL, \
      `airblock_id` INT(11) NULL DEFAULT NULL, \
      `lower_fl` INT(11) NULL DEFAULT NULL, \
      `upper_fl` INT(11) NULL DEFAULT NULL, \
      `operation` VARCHAR(1) NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      INDEX `airblock_sector_link` (`airblock_id` ASC), \
      INDEX `sector_airblock_link` (`sector_id` ASC), \
      CONSTRAINT `airblock_sector_link` \
        FOREIGN KEY (`airblock_id`) \
        REFERENCES "+db_name+".`airblock` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `sector_airblock_link` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    AUTO_INCREMENT = 262031 \
    DEFAULT CHARACTER SET = latin1")


    # Table trafficvolume_has_airport
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_airport`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_airport` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `airport_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`trafficvolume_id`, `airport_id`), \
      INDEX `fk_TrafficVolume_has_Airport_Airport1` (`airport_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_Airport_Airport1` \
        FOREIGN KEY (`airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_Airport_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    # Table trafficvolume_has_airportset
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_airportset`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_airportset` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `airportset_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`trafficvolume_id`, `airportset_id`), \
      INDEX `fk_TrafficVolume_has_AirportSet_AirportSet1` (`airportset_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_AirportSet_AirportSet1` \
        FOREIGN KEY (`airportset_id`) \
        REFERENCES "+db_name+".`airportset` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_AirportSet_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")


    #Table trafficvolume_has_airspace
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_airspace` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `airspace_id` INT(11) NOT NULL, \
      PRIMARY KEY (`trafficvolume_id`, `airspace_id`), \
      INDEX `fk_TrafficVolume_has_Airspace_Airspace1` (`airspace_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_Airspace_Airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_Airspace_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table trafficvolume_has_flow

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_flow`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_flow` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `flow_id` INT(11) NOT NULL, \
      `type` VARCHAR(2) NOT NULL, \
      PRIMARY KEY (`trafficvolume_id`, `flow_id`, `type`), \
      INDEX `fk_TrafficVolume_has_Flow_Flow1` (`flow_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_Flow_Flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_Flow_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`trafficvolume_has_sector`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_sector` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `sector_id` INT(11) NOT NULL, \
      PRIMARY KEY (`trafficvolume_id`, `sector_id`), \
      INDEX `fk_TrafficVolume_has_Sector_Sector1` (`sector_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_Sector_Sector1` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_Sector_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`trafficvolume_has_waypoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trafficvolume_has_waypoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trafficvolume_has_waypoint` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `waypoint_id` INT(11) NOT NULL, \
      `min_fl` INT(11) NULL DEFAULT NULL, \
      `max_fl` INT(11) NULL DEFAULT NULL, \
      `sense` CHAR(1) NULL, \
      PRIMARY KEY (`trafficvolume_id`, `waypoint_id`), \
      INDEX `fk_TrafficVolume_has_WayPoint_WayPoint1` (`waypoint_id` ASC), \
      CONSTRAINT `fk_TrafficVolume_has_WayPoint_TrafficVolume1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_TrafficVolume_has_WayPoint_WayPoint1` \
        FOREIGN KEY (`waypoint_id`) \
        REFERENCES "+db_name+".`waypoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`geopoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`geopoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`geopoint` ( \
      `id` INT NOT NULL AUTO_INCREMENT, \
      `coords` POINT NOT NULL, \
      `type` VARCHAR(3) NULL, \
      PRIMARY KEY (`id`), \
      INDEX `type_indx` (`type` ASC)) \
    ENGINE = InnoDB")
    #INDEX `coord_indx` (`coords` ASC), \



    #Table "+db_name+".`waypoint_has_geopoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`waypoint_has_geopoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`waypoint_has_geopoint` ( \
      `waypoint_id` INT(11) NOT NULL, \
      `geopoint_id` INT NOT NULL, \
      `airac` INT NOT NULL, \
      `name` VARCHAR(50) NULL, \
      PRIMARY KEY (`waypoint_id`, `geopoint_id`, `airac`), \
      INDEX `fk_waypoint_has_point_point1_idx` (`geopoint_id` ASC), \
      INDEX `fk_waypoint_has_point_waypoint1_idx` (`waypoint_id` ASC), \
      CONSTRAINT `fk_waypoint_has_point_waypoint1` \
        FOREIGN KEY (`waypoint_id`) \
        REFERENCES "+db_name+".`waypoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_waypoint_has_point_geopoint1` \
        FOREIGN KEY (`geopoint_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`airport_has_geopoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airport_has_geopoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airport_has_geopoint` ( \
      `airport_id` INT(11) NOT NULL AUTO_INCREMENT, \
      `geopoint_id` INT NOT NULL, \
      `airac` INT NOT NULL, \
      `name` VARCHAR(45) NULL, \
      `altitude` INT NULL, \
      `tis` INT NULL, \
      `trs` INT NULL, \
      `taxi_time` INT NULL, \
      PRIMARY KEY (`airport_id`, `geopoint_id`, `airac`), \
      INDEX `fk_airport_has_point_point1_idx` (`geopoint_id` ASC), \
      INDEX `fk_airport_has_point_airport1_idx` (`airport_id` ASC), \
      CONSTRAINT `fk_airport_has_point_airport1` \
        FOREIGN KEY (`airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_airport_has_point_geopoint1` \
        FOREIGN KEY (`geopoint_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`configuration_has_airspace`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`configuration_has_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`configuration_has_airspace` ( \
      `configuration_id` INT(11) NOT NULL, \
      `airspace_id` INT(11) NOT NULL, \
      PRIMARY KEY (`configuration_id`, `airspace_id`), \
      INDEX `fk_configuration_has_airspace_airspace1_idx` (`airspace_id` ASC), \
      INDEX `fk_configuration_has_airspace_configuration1_idx` (`configuration_id` ASC), \
      CONSTRAINT `fk_configuration_has_airspace_configuration1` \
        FOREIGN KEY (`configuration_id`) \
        REFERENCES "+db_name+".`configuration` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_configuration_has_airspace_airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`flow_has_airspace`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`flow_has_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`flow_has_airspace` ( \
      `flow_id` INT(11) NOT NULL, \
      `airspace_id` INT(11) NOT NULL, \
      `sense` CHAR(1) NOT NULL, \
      PRIMARY KEY (`flow_id`, `airspace_id`, `sense`), \
      INDEX `fk_flow_has_airspace_airspace1_idx` (`airspace_id` ASC), \
      INDEX `fk_flow_has_airspace_flow1_idx` (`flow_id` ASC), \
      CONSTRAINT `fk_flow_has_airspace_flow1` \
        FOREIGN KEY (`flow_id`) \
        REFERENCES "+db_name+".`flow` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_flow_has_airspace_airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`airportset_has_airport`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`airportset_has_airport`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`airportset_has_airport` ( \
      `airportset_id` INT(11) NOT NULL, \
      `airport_id` INT(11) NOT NULL, \
      `airac` VARCHAR(45) NOT NULL, \
      `name` VARCHAR(50) NULL, \
      `type` VARCHAR(4) NULL, \
      PRIMARY KEY (`airportset_id`, `airport_id`, `airac`), \
      INDEX `fk_airportset_has_airport_airport1_idx` (`airport_id` ASC), \
      INDEX `fk_airportset_has_airport_airportset1_idx` (`airportset_id` ASC), \
      CONSTRAINT `fk_airportset_has_airport_airportset1` \
        FOREIGN KEY (`airportset_id`) \
        REFERENCES "+db_name+".`airportset` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_airportset_has_airport_airport1` \
        FOREIGN KEY (`airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB \
    DEFAULT CHARACTER SET = latin1")



    #Table "+db_name+".`trajectory`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory` ( \
      `id` INT NOT NULL AUTO_INCREMENT, \
      `flight_id` INT(11) NOT NULL, \
      `type` VARCHAR(4) NOT NULL, \
      `airac_cycle_release` INT NULL, \
      `env_baseline_number` INT NULL DEFAULT NULL, \
      `departure_runway` VARCHAR(15) NULL DEFAULT NULL, \
      `arrival_runway` VARCHAR(15) NULL DEFAULT NULL, \
      `fuel_consumed` DECIMAL(20,2) NULL DEFAULT NULL, \
      `crco` DECIMAL(12,2) NULL DEFAULT NULL, \
      `obt` DATETIME NULL DEFAULT NULL, \
      PRIMARY KEY (`id`), \
      INDEX `fk_trajectory_flight_trajectories1_idx` (`flight_id` ASC), \
      INDEX `index_type` (`type` ASC), \
      CONSTRAINT `fk_trajectory_flight_trajectories1` \
        FOREIGN KEY (`flight_id`) \
        REFERENCES "+db_name+".`flight` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    #Table "+db_name+".`coordpoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`coordpoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`coordpoint` ( \
      `id` INT NOT NULL AUTO_INCREMENT, \
      `sid` VARCHAR(60) NOT NULL, \
      PRIMARY KEY (`id`), \
      UNIQUE INDEX `sid_UNIQUE` (`sid` ASC)) \
    ENGINE = InnoDB")



    #Table "+db_name+".`coordpoint_has_geopoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`coordpoint_has_geopoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`coordpoint_has_geopoint` ( \
      `coordpoint_id` INT NOT NULL, \
      `geopoint_id` INT NOT NULL, \
      PRIMARY KEY (`coordpoint_id`, `geopoint_id`), \
      INDEX `fk_coordpoint_has_geopoint_geopoint1_idx` (`geopoint_id` ASC), \
      INDEX `fk_coordpoint_has_geopoint_coordpoint1_idx` (`coordpoint_id` ASC), \
      CONSTRAINT `fk_coordpoint_has_geopoint_coordpoint1` \
        FOREIGN KEY (`coordpoint_id`) \
        REFERENCES "+db_name+".`coordpoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_coordpoint_has_geopoint_geopoint1` \
        FOREIGN KEY (`geopoint_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")



    #Table "+db_name+".`trajectory_has_geopoint`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_has_geopoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_has_geopoint` ( \
      `trajectory_id` INT NOT NULL, \
      `geopoint_id` INT NOT NULL, \
      `distance` INT NULL, \
      `fl` INT NULL, \
      `time_over` DATETIME NULL, \
      `type` VARCHAR(5) NULL, \
      `rel_dist` INT NULL, \
      `visible` TINYINT(1) NULL, \
      `order` INT NULL, \
      INDEX `fk_trajectory_has_geopoint_geopoint1_idx` (`geopoint_id` ASC), \
      CONSTRAINT `fk_trajectory_has_geopoint_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_geopoint_geopoint1` \
        FOREIGN KEY (`geopoint_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")



    #Table "+db_name+".`trajectory_has_sector`
    #GEOPOINT_ENTRY AND EXIT WHERE NOT NULL

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_has_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_has_sector` ( \
      `trajectory_id` INT NOT NULL, \
      `sector_id` INT(11) NOT NULL, \
      `geopoint_entry_id` INT NULL, \
      `geopoint_exit_id` INT NULL, \
      `fl_entry` INT NULL, \
      `fl_exit` INT NULL, \
      `distance_entry` INT NULL, \
      `distance_exit` INT NULL, \
      `time_entry` DATETIME NULL, \
      `time_exit` DATETIME NULL, \
      `order` INT NULL, \
      INDEX `fk_trajectory_has_sector_trajectory1_idx` (`trajectory_id` ASC), \
      INDEX `fk_trajectory_has_sector_sector1_idx` (`sector_id` ASC), \
      INDEX `fk_trajectory_has_sector_geopoint1_idx` (`geopoint_entry_id` ASC), \
      INDEX `fk_trajectory_has_sector_geopoint2_idx` (`geopoint_exit_id` ASC), \
      CONSTRAINT `fk_trajectory_has_sector_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_sector_sector1` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_sector_geopoint1` \
        FOREIGN KEY (`geopoint_entry_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_sector_geopoint2` \
        FOREIGN KEY (`geopoint_exit_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")



    #Table "+db_name+".`trajectory_has_airspace`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_has_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_has_airspace` ( \
      `trajectory_id` INT NOT NULL, \
      `airspace_id` INT(11) NOT NULL, \
      `geopoint_entry_id` INT NULL, \
      `geopoint_exit_id` INT NULL, \
      `fl_entry` INT NULL, \
      `fl_exit` INT NULL, \
      `distance_entry` INT NULL, \
      `distance_exit` INT NULL, \
      `time_entry` DATETIME NULL, \
      `time_exit` DATETIME NULL, \
      `order` INT NULL, \
      INDEX `fk_trajectory_has_airspace_trajectory1_idx` (`trajectory_id` ASC), \
      INDEX `fk_trajectory_has_airspace_airspace1_idx` (`airspace_id` ASC), \
      INDEX `fk_trajectory_has_airspace_geopoint1_idx` (`geopoint_entry_id` ASC), \
      INDEX `fk_trajectory_has_airspace_geopoint2_idx` (`geopoint_exit_id` ASC), \
      CONSTRAINT `fk_trajectory_has_airspace_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_airspace_airspace1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_airspace_geopoint1` \
        FOREIGN KEY (`geopoint_entry_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_airspace_geopoint2` \
        FOREIGN KEY (`geopoint_exit_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")



    #Table "+db_name+".`trajectory_has_circle_intersections`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_has_circle_intersections`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_has_circle_intersections` ( \
      `trajectory_id` INT NOT NULL, \
      `type` VARCHAR(4) NOT NULL, \
      `from_last` VARCHAR(1) NULL, \
      `to_from_airport_id` INT(11) NOT NULL, \
      `geopoint_entry_id` INT NULL, \
      `geopoint_exit_id` INT NULL, \
      `fl_entry` INT NULL, \
      `fl_exit` INT NULL, \
      `distance_radius_nm` INT NULL, \
      `time_entry` DATETIME NULL, \
      `time_exit` DATETIME NULL, \
      `distance_entry` INT NULL, \
      `distance_exit` INT NULL, \
      INDEX `fk_trajectory_has_circle_intersections_airport1_idx` (`to_from_airport_id` ASC), \
      INDEX `fk_trajectory_has_circle_intersections_geopoint1_idx` (`geopoint_entry_id` ASC), \
      INDEX `fk_trajectory_has_circle_intersections_geopoint2_idx` (`geopoint_exit_id` ASC), \
      CONSTRAINT `fk_trajectory_has_circle_intersections_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_circle_intersections_airport1` \
        FOREIGN KEY (`to_from_airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_circle_intersections_geopoint1` \
        FOREIGN KEY (`geopoint_entry_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_trajectory_has_circle_intersections_geopoint2` \
        FOREIGN KEY (`geopoint_exit_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")



    #Table "+db_name+".`trajectory_fl_request`

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_fl_request`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_fl_request` ( \
      `trajectory_id` INT NOT NULL, \
      `fl` INT NULL, \
      `speed` INT NULL, \
      `speed_units` VARCHAR(1) NULL, \
      `distance` INT NULL, \
      INDEX `fk_fl_request_trajectory1_idx` (`trajectory_id` ASC), \
      CONSTRAINT `fk_fl_request_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    # Table trajectory_eet_fir

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_eet_fir`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_eet_fir` ( \
      `trajectory_id` INT NOT NULL, \
      `fir_sid` varchar(45) NULL, \
      `fir_id` INT(11) NULL, \
      `eet` INT NULL, \
      `order` INT NULL, \
      INDEX `fk_eet_fir_trajectory1_idx` (`trajectory_id` ASC), \
      INDEX `fk_eet_fir_sector1_idx` (`fir_id` ASC), \
      CONSTRAINT `fk_eet_fir_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_eet_fir_sector1` \
        FOREIGN KEY (`fir_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    #Table "+db_name+".`trajectory_eet_point` 
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`trajectory_eet_point`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`trajectory_eet_point` ( \
      `trajectory_id` INT NOT NULL, \
      `geopoint_id` INT NULL, \
      `eet` INT NULL, \
      `order` INT NULL, \
      `point_sid` VARCHAR(15) NULL, \
      INDEX `fk_eet_points_trajectory1_idx` (`trajectory_id` ASC), \
      INDEX `fk_eet_points_geopoint1_idx` (`geopoint_id` ASC), \
      CONSTRAINT `fk_eet_points_trajectory1` \
        FOREIGN KEY (`trajectory_id`) \
        REFERENCES "+db_name+".`trajectory` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION, \
      CONSTRAINT `fk_eet_points_geopoint1` \
        FOREIGN KEY (`geopoint_id`) \
        REFERENCES "+db_name+".`geopoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    #Table capacity_sector
    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_sector`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_sector` ( \
      `sector_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_sector_id1_idx` (`sector_id` ASC), \
      CONSTRAINT `fk_sector_id1` \
        FOREIGN KEY (`sector_id`) \
        REFERENCES "+db_name+".`sector` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_airspace`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_airspace` ( \
      `airspace_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_airspace_id1_idx` (`airspace_id` ASC), \
      CONSTRAINT `fk_airspace_id1` \
        FOREIGN KEY (`airspace_id`) \
        REFERENCES "+db_name+".`airspace` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_airport`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_airport` ( \
      `airport_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_airport_id1_idx` (`airport_id` ASC), \
      CONSTRAINT `fk_airport_id1` \
        FOREIGN KEY (`airport_id`) \
        REFERENCES "+db_name+".`airport` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")

    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_airportset`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_airportset` ( \
      `airportset_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_airportset_id1_idx` (`airportset_id` ASC), \
      CONSTRAINT `fk_airportset_id1` \
        FOREIGN KEY (`airportset_id`) \
        REFERENCES "+db_name+".`airportset` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_trafficvolume`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_trafficvolume` ( \
      `trafficvolume_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_trafficvolume_id1_idx` (`trafficvolume_id` ASC), \
      CONSTRAINT `fk_trafficvolume_id1` \
        FOREIGN KEY (`trafficvolume_id`) \
        REFERENCES "+db_name+".`trafficvolume` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    conn.execute("DROP TABLE IF EXISTS "+db_name+".`capacity_waypoint`")

    conn.execute("CREATE TABLE IF NOT EXISTS "+db_name+".`capacity_waypoint` ( \
      `waypoint_id` INT(11) NOT NULL, \
      `start` DATETIME NOT NULL, \
      `end` DATETIME NOT NULL, \
      `capacity` INT(11) NOT NULL, \
      `hourly3capacity` INT(11), \
      `category` CHAR(1) NOT NULL, \
      `env_table_source` CHAR(1), \
      INDEX `fk_waypoint_id1_idx` (`waypoint_id` ASC), \
      CONSTRAINT `fk_waypoint_id1` \
        FOREIGN KEY (`waypoint_id`) \
        REFERENCES "+db_name+".`waypoint` (`id`) \
        ON DELETE NO ACTION \
        ON UPDATE NO ACTION) \
    ENGINE = InnoDB")


    #Create views
    conn.execute("DROP VIEW IF EXISTS "+db_name+".`airspace_sectors`")

    conn.execute("CREATE VIEW "+db_name+".`airspace_sectors` AS \
        SELECT  \
    `a`.`id` AS `aid`, \
    `a`.`airac` AS `airac`, \
    `a`.`sid` AS `asid`, \
    `a`.`type` AS `type`, \
    `a`.`name` AS `name`, \
    `s`.`id` AS `sid`, \
    `s`.`sid` AS `ssid`, \
    `s`.`name` AS `sname`, \
    `s`.`type` AS `stype`, \
    `s`.`category` AS `scategory` \
        FROM \
            (("+db_name+".`airspace` `a` \
            LEFT JOIN "+db_name+".`airspace_has_sector` `ahs` ON ((`a`.`id` = `ahs`.`airspace_id`))) \
            JOIN "+db_name+".`sector` `s` ON ((`ahs`.`sector_id` = `s`.`id`))) \
        UNION SELECT \
            `a`.`id` AS `aid`, \
            `a`.`airac` AS `airac`, \
            `a`.`sid` AS `asid`, \
            `a`.`type` AS `type`, \
            `a`.`name` AS `name`, \
            `s2`.`id` AS `sid`, \
            `s2`.`sid` AS `ssid`, \
            `s2`.`name` AS `sname`, \
            `s2`.`type` AS `stype`, \
            `s2`.`category` AS `scategory` \
        FROM \
            (((("+db_name+".`airspace` `a` \
            LEFT JOIN "+db_name+".`airspace_has_airspace` `aha` ON ((`a`.`id` = `aha`.`parent_airspace_id`))) \
            LEFT JOIN "+db_name+".`airspace` `a2` ON ((`a2`.`id` = `aha`.`subairspace_id`))) \
            LEFT JOIN "+db_name+".`airspace_has_sector` `ahs2` ON ((`a2`.`id` = `ahs2`.`airspace_id`))) \
            JOIN "+db_name+".`sector` `s2` ON ((`ahs2`.`sector_id` = `s2`.`id`)))")


    conn.execute("DROP VIEW IF EXISTS "+db_name+".`airspace_sectors_airblock`")

    conn.execute("CREATE VIEW "+db_name+".`airspace_sectors_airblock` AS \
        SELECT \
            `a`.`id` AS `aid`, \
            `a`.`airac` AS `airac`, \
            `a`.`sid` AS `asid`, \
            `a`.`type` AS `type`, \
            `a`.`name` AS `name`, \
            `s`.`id` AS `sid`, \
            `s`.`sid` AS `ssid`, \
            `s`.`name` AS `sname`, \
            `s`.`type` AS `stype`, \
            `s`.`category` AS `scategory`, \
            `ss`.`lower_fl` AS `lower_fl`, \
            `ss`.`upper_fl` AS `upper_fl`, \
            `ss`.`operation` AS `operation`, \
            `ab`.`name` AS `abname`, \
            `ab`.`boundary` AS `boundary` \
        FROM \
            (((("+db_name+".`airspace` `a` \
            LEFT JOIN "+db_name+".`airspace_has_sector` `ahs` ON ((`a`.`id` = `ahs`.`airspace_id`))) \
            JOIN "+db_name+".`sector` `s` ON ((`ahs`.`sector_id` = `s`.`id`))) \
            JOIN "+db_name+".`sectorslice` `ss` ON ((`ss`.`sector_id` = `s`.`id`))) \
            JOIN "+db_name+".`airblock` `ab` ON ((`ss`.`airblock_id` = `ab`.`id`))) \
        UNION SELECT \
            `a`.`id` AS `aid`, \
            `a`.`airac` AS `airac`, \
            `a`.`sid` AS `asid`, \
            `a`.`type` AS `type`, \
            `a`.`name` AS `name`, \
            `s2`.`id` AS `sid`, \
            `s2`.`sid` AS `ssid`, \
            `s2`.`name` AS `sname`, \
            `s2`.`type` AS `stype`, \
            `s2`.`category` AS `scategory`, \
            `ss`.`lower_fl` AS `lower_fl`, \
            `ss`.`upper_fl` AS `upper_fl`, \
            `ss`.`operation` AS `operation`, \
            `ab`.`name` AS `abname`, \
            `ab`.`boundary` AS `boundary` \
        FROM \
            (((((("+db_name+".`airspace` `a` \
            LEFT JOIN "+db_name+".`airspace_has_airspace` `aha` ON ((`a`.`id` = `aha`.`parent_airspace_id`))) \
    LEFT JOIN "+db_name+".`airspace` `a2` ON ((`a2`.`id` = `aha`.`subairspace_id`))) \
    LEFT JOIN "+db_name+".`airspace_has_sector` `ahs2` ON ((`a2`.`id` = `ahs2`.`airspace_id`))) \
    JOIN "+db_name+".`sector` `s2` ON ((`ahs2`.`sector_id` = `s2`.`id`))) \
    JOIN "+db_name+".`sectorslice` `ss` ON ((`ss`.`sector_id` = `s2`.`id`))) \
    JOIN "+db_name+".`airblock` `ab` ON ((`ss`.`airblock_id` = `ab`.`id`)))")


    conn.execute("DROP VIEW IF EXISTS "+db_name+".`trafficvolume_with_parts`")

    conn.execute("CREATE VIEW "+db_name+".`trafficvolume_with_parts` AS \
        SELECT \
            `tv`.`id` AS `id`, \
            `tv`.`airac` AS `airac`, \
            `tv`.`sid` AS `sid_tv`, \
            `tv`.`name` AS `name_tv`, \
            `tv`.`category` AS `category_tv`, \
            `tva`.`airport_id` AS `airport_id`, \
            `tva`.`sense` AS `senseAirport`, \
            `tvas`.`airportset_id` AS `airportset_id`, \
            `tvas`.`sense` AS `sense_airportset`, \
            `tvair`.`airspace_id` AS `airspace_id`, \
            `tvs`.`sector_id` AS `sector_id`, \
            `tvf`.`flow_id` AS `flow_id`, \
            `tvf`.`type` AS `flow_type`, \
            `tvwp`.`waypoint_id` AS `waypoint_id`, \
            `tvwp`.`min_fl` AS `waypoint_min_fl`, \
            `tvwp`.`max_fl` AS `waypoint_max_fl` \
        FROM \
            (((((("+db_name+".`trafficvolume` `tv` \
            LEFT JOIN "+db_name+".`trafficvolume_has_airport` `tva` ON ((`tva`.`trafficvolume_id` = `tv`.`id`))) \
            LEFT JOIN "+db_name+".`trafficvolume_has_airportset` `tvas` ON ((`tvas`.`trafficvolume_id` = `tv`.`id`))) \
            LEFT JOIN "+db_name+".`trafficvolume_has_airspace` `tvair` ON ((`tvair`.`trafficvolume_id` = `tv`.`id`))) \
            LEFT JOIN "+db_name+".`trafficvolume_has_sector` `tvs` ON ((`tvs`.`trafficvolume_id` = `tv`.`id`))) \
            LEFT JOIN "+db_name+".`trafficvolume_has_flow` `tvf` ON ((`tvf`.`trafficvolume_id` = `tv`.`id`))) \
            LEFT JOIN "+db_name+".`trafficvolume_has_waypoint` `tvwp` ON ((`tvwp`.`trafficvolume_id` = `tv`.`id`)))")


    conn.execute("SET SQL_MODE=@OLD_SQL_MODE")
    conn.execute("SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS")
    conn.execute("SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS")

def file2db_airac_info(engine,airac,fpath='.',filename='OpeningScheme_'):
    if filename.endswith('.cos'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.cos'

    data = pd.read_csv(filename, sep=";", skiprows=1, header=None)
    data.columns = ['date', 'scheme','hini','hend','conf','origin']
       
    data['date']=data['date'].apply(format_date)
    
    min_date=data.iloc[0,0]
    max_date=data.iloc[len(data.index)-1,0]
    
    airac_data=str(airac)+",\'"+str(min_date)+"\',\'"+str(max_date)+"\'"
    
    sql = "INSERT INTO airac (id,date_start,date_end) VALUES ("+airac_data+")"

    engine.execute(sql)

def file2db_airports(engine,airac,fpath='.',filename='Airport_',ensure_weird=False):
    if filename.endswith('.narp'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.narp'

    data = pd.read_csv(filename, sep=";", skiprows=1, header=None)
    data.columns = ['sid', 'name','lat','lon','tis','trs','taxi_time','altitude']

    data["latlon"] = data["lat"].map(str) +"_"+ data["lon"].map(str)

    geopoints=read_geopoints_ids(engine)

    missing_geopoints=set(data["latlon"])-geopoints.keys()

    if len(missing_geopoints)>0:
        d=pd.DataFrame(list(missing_geopoints))
        d.columns=['latlon']
        d['type']="A"
        add_geopoints(engine,d)
        geopoints=read_geopoints_ids(engine)

    airports=read_airports_id(engine)

    missing_airports=set(data["sid"])-airports.keys()

    if len(missing_airports)>0:
        add_airports(engine,missing_airports)
        airports=read_airports_id(engine)

    data['airport_id']=data['sid'].apply(lambda x: airports.get(x))
    data['geopoint_id']=data['latlon'].apply(lambda x: geopoints.get(x)['id'])
    data['altitude']=data['altitude'].apply(lambda x: x if x != "_" else np.nan)
    data['tis']=data['tis'].apply(lambda x: x if x != "_" else np.nan)
    data['trs']=data['trs'].apply(lambda x: x if x != "_" else np.nan)
    data['taxi_time']=data['taxi_time'].apply(lambda x: x if x != "_" else np.nan)
    data['airac']=airac

    add_airport_point_connection(engine,data)

    if ensure_weird:
        #Check that ZZZZ and AFIL are in the list of airports, otherwise add them
        airports=read_airports_id(engine)

        if airports.get("ZZZZ",None)==None:
            #insert ZZZZ
            add_airports(engine,["ZZZZ"])

        if airports.get("AFIL",None)==None:
            #insert AFIL
            add_airports(engine,["AFIL"])

def file2db_set_of_airports(engine,airac,fpath='.',filename='SetOfAirports_'):
    if filename.endswith('.nsoa'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.nsoa'

    data = pd.read_csv(filename, sep=";", skiprows=1, header=None, names=["SA","sid","name","type","nelems"])
    
    airports_id=read_airports_id(engine)

    data.loc[data['SA']=="A", "airport_id"]=data["sid"].apply(lambda x: airports_id.get(x,{}))

    as_dict=read_airport_sets_ids(engine)
    missing_airportsets=set(data.loc[data['SA']=="S"]["sid"])-as_dict.keys()

    if len(missing_airportsets)>0:
        d=pd.DataFrame(list(missing_airportsets))
        d.columns=['sid']
        add_airportsets(engine,d)
        as_dict=read_airport_sets_ids(engine)

    data.loc[data['SA']=="S", "airportset_id"]=data["sid"].apply(lambda x: as_dict.get(x,{}))

    data.loc[data['SA']=="A", "airac"]=airac

    C=[]

    asa=0
    for r in data['SA']:
        if r[0]=="S":
            asa=asa+1
        C.append(asa)

    data['asan'] = C

    df=data.loc[(data.loc[:,'SA']=="S"),["asan","name","type","airportset_id"]]
    dict_name=pd.Series(df.name.values,index=df.asan).to_dict()
    dict_type=pd.Series(df.type.values,index=df.asan).to_dict()
    dict_asid=pd.Series(df.airportset_id.values,index=df.asan).to_dict()

    data["name"]=data["asan"].apply(lambda x: dict_name.get(x))
    data["type"]=data["asan"].apply(lambda x: dict_type.get(x))
    data["airportset_id"]=data["asan"].apply(lambda x: dict_asid.get(x))
    
    add_airportset_has_airport(engine,data.loc[data.loc[:,'SA']=="A",:])

def file2db_navpoints(engine,airac,fpath='.',filename='NavPoint_'):
    if filename.endswith('.nnpt'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.nnpt'
    
    data = pd.read_csv(filename, sep=";", skiprows=1, header=None,na_filter=False)
    data.columns = ['sid', 'type','lat','lon','name']

    data["latlon"] = data["lat"].map(str) +"_"+ data["lon"].map(str)
    data['sid']=data['sid'].map(str)
    data['airac']=airac

    geopoints=read_geopoints_ids(engine)

    missing_geopoints=set(data["latlon"])-geopoints.keys()

    if len(missing_geopoints)>0:
        d=pd.DataFrame(list(missing_geopoints))
        d.columns=['latlon']
        d['type']="DB"
        add_geopoints(engine,d)
        geopoints=read_geopoints_ids(engine)

    wpt_dict=read_waypoints_ids(engine)

    missing_wpt=set(data["sid"])-wpt_dict.keys()
    if len(missing_wpt)>0:
        add_waypoint(engine,data.loc[data['sid'].isin(missing_wpt)])    
        wpt_dict=read_waypoints_ids(engine)

    data['waypoint_id']=data['sid'].apply(lambda x: wpt_dict.get(x))
    data['geopoint_id']=data['latlon'].apply(lambda x: geopoints.get(x)['id'])

    add_waypoint_point_connection(engine,data)

def file2db_airblocks(engine,airac,fpath='.',filename='sectors_'):
    if filename.endswith('.gar'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.gar'

    airblocks = {}

    with open(filename) as f:
        lines = f.readlines()

    lines= [x.strip() for x in lines]

    for i in range(0,len(lines)):
        lsp=lines[i].split(";")
        if lsp[0]=="A":
            name=lsp[1]
        elif lsp[0]=="P":
            lat=lsp[1]
            lon=lsp[2]

            airblocks[name]=(airblocks.get(name,""))+", "+lat+" "+lon

    data = pd.DataFrame(list(airblocks.items()), columns=['name', 'poly'])
    data['poly']=data['poly'].apply(lambda x: "POLYGON (( "+x[1:len(x)]+" ))")
    data['airac']=airac
    
    add_airblocks(engine,data)

def file2db_sectors_sectorslices(engine,airac,fpath='.',filename='sectors_'):   
    if filename.endswith('.gsl'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.gsl'
  
    ab_dict = read_airblocks_ids(engine,airac)
    
    sectors={}
    ab={}

    with open(filename) as f:
        lines = f.readlines()

    lines= [x.strip() for x in lines]

    for i in range(0,len(lines)):
        lsp=lines[i].split(";")
        if lsp[0]=="S":
            sid=lsp[1]
            name=lsp[2]
            category=lsp[4] #could be non existent
            types=lsp[5]
            sectors[sid]=[name, category, types]
        elif lsp[0]=="A":
            name=lsp[1]
            abid=ab_dict.get(name)
            operation=lsp[2]
            minAlt=int(lsp[3])
            maxAlt=int(lsp[4])
            ab[i]=[sid,abid,name,operation,minAlt,maxAlt]
            

    data_sectors = pd.DataFrame(sectors).transpose()
    data_sectors.reset_index(level=0, inplace=True)
    data_sectors.columns=['sid','name','category','type']
    data_sectors['airac']=airac
    add_sectors(engine,data_sectors)
    
    s_dict=read_sectors_ids(engine,airac)

    data_sectorslice = pd.DataFrame(ab).transpose()
    data_sectorslice.reset_index(level=0, inplace=True)
    data_sectorslice.columns=['index','sector_sid','airblock_id','airblock_name','operation','lower_fl','upper_fl']
    
    data_sectorslice['sector_id']=data_sectorslice['sector_sid'].apply(lambda x: s_dict.get(x))
    
    add_sectorslice(engine,data_sectorslice)

def file2db_airspace(engine,airac,fpath='.',filename='Airspace_'):
    if filename.endswith('.spc'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.spc'

    sec_dict=read_sectors_ids(engine,airac)
    
    airspace={}
    sub_structures={}
    
    airspace_codes=["AREA", "NAS", "AUAG", "AUA", "CLUS", "CS", "CRSA"]
    sector_codes=["ES", "FIR", "ERSA"]

    with open(filename) as f:
        lines = f.readlines()

    lines= [x.strip() for x in lines]

    for i in range(0,len(lines)):
        lsp=lines[i].split(";")
        if lsp[0]=="A":
            sid=lsp[1];
            name=lsp[2]
            typea=lsp[3]
            airspace[sid]=[name, typea]
            
        elif lsp[0]=="S":
            name=lsp[1]
            types=lsp[2]
            
            if types in airspace_codes:
                a="sairspace_"+name
            elif types in sector_codes:
                a="ssector_"+name
            else:
                a="error_code"+types
            
            sub_structures[i]=[a,sid,types]
            
    data_airspace = pd.DataFrame(airspace).transpose()
    data_airspace.reset_index(level=0, inplace=True)
    data_airspace.columns=['sid','name','type']       
    data_airspace['airac']=airac
    
    add_airspace(engine,data_airspace)
    
    airs_dic=read_airspace_ids(engine,airac)
    
    data_subairspace = pd.DataFrame(sub_structures).transpose()
    data_subairspace.reset_index(level=0, inplace=True)
    data_subairspace.columns=['index','satype','name','type']       
    data_subairspace['airac']=airac
    data_subairspace['sub_type']=data_subairspace['satype'].apply(lambda x: x.split("_")[0])
    data_subairspace['sub_id']=data_subairspace['satype'].\
        apply(lambda x: airs_dic.get(x.split("_")[1]) if x.split("_")[0]=="sairspace" \
              else sec_dict.get(x.split("_")[1],"A"))
    data_subairspace['super_type']=data_subairspace['name'].apply(lambda x: airs_dic.get(x))
    
    sairspace=data_subairspace.loc[data_subairspace.loc[:,'sub_type']=="sairspace",['sub_id', 'super_type']]
    sairspace.columns=['subairspace_id','parent_airspace_id']
    
    ssectors=data_subairspace.loc[data_subairspace.loc[:,'sub_type']=="ssector",['sub_id', 'super_type']]
    ssectors.columns=['sector_id','airspace_id']
      
    add_airspace_has_airspace(engine,sairspace)
    add_airspace_has_sector(engine,ssectors)

def file2db_controllers_availability(engine,airac,fpath='.',filename='ControllerAvailability_'):
    if filename.endswith('.nbak'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.nbak'
    
    a_dict=read_airspace_ids(engine,airac)
    
    data = pd.read_csv(filename, sep=";", skiprows=0, header=None)
    data.columns = ['a_sid', 'datenf','hstart','hend','max_controllers_pos']
    data['date']=data['datenf'].apply(lambda x: x.split("/")[2]+"-"+x.split("/")[1]+"-"+x.split("/")[0])
    data['start']=data.apply(lambda x: x['date']+" "+x['hstart']+":00", axis=1)
    data['end']=data.apply(lambda x: x['date']+" "+x['hend']+":00", axis=1)
    data['airac']=airac
    data['airspace_id']=data['a_sid'].apply(lambda x: a_dict.get(x))
    
    add_controllers_availability(engine,data)

def file2db_configurations(engine,airac,fpath='.',filenameControllers='ConfNbController_',filenameConfig='Configuration_'):
    if filenameControllers is not None:
      if filenameControllers.endswith('.ncnc'):
          filename_controller=fpath+'/'+filenameControllers
      else:
          filename_controller=fpath+'/'+filenameControllers+str(airac)+'.ncnc'

    if filenameConfig.endswith('.cfg'):
      filename_config=fpath+'/'+filenameConfig
      skiprows_cfg = 1
    else:
      filename_config=fpath+'/'+filenameConfig+str(airac)+'.ncnc'
      skiprows_cfg = 0


    a_dict=read_airspace_ids(engine,airac)
    s_dict=read_sectors_ids(engine,airac)

    if filenameControllers is not None:
      data_ncontrollers = pd.read_csv(filename_controller, sep=";", skiprows=0, header=None)
      if len(data_ncontrollers)>0:
        data_ncontrollers.columns = ['acc_sid', 'conf_sid','num_controllers']
        data_ncontrollers['acc_conf']=data_ncontrollers.apply(lambda x: x['acc_sid']+"_"+x['conf_sid'], axis=1)
        numcont_dict=data_ncontrollers.set_index('acc_conf').to_dict()['num_controllers']
      else:
        numcont_dict = {}
    else:
      numcont_dict = {}

    data_configurations = pd.read_csv(filename_config, sep=";", skiprows=skiprows_cfg, header=None)
    data_configurations.columns = ['acc_sid', 'sid','sect_sid']
    data_configurations['acc_conf']=data_configurations.apply(lambda x: x['acc_sid']+"_"+x['sid'], axis=1)
    data_configurations['num_controllers']=data_configurations['acc_conf'].apply(lambda x: numcont_dict.get(x))
    data_configurations['acc_id']=data_configurations['acc_sid'].apply(lambda x: a_dict.get(x))
    data_configurations['sector_id']=data_configurations['sect_sid'].apply(lambda x: s_dict.get(x))
    data_configurations['airspace_id']=data_configurations['sect_sid'].apply(lambda x: a_dict.get(x))
    data_configurations['airac']=airac

    add_configuration(engine,data_configurations[['sid','num_controllers','acc_id','airac']].drop_duplicates())
    conf_dict=read_configurations_ids(engine,airac)

    data_configurations["configuration_id"]=data_configurations["acc_conf"].apply(lambda x: conf_dict.get(x))

    add_configuration_has_sector(engine,data_configurations.loc[~pd.isnull(data_configurations.loc[:,'sector_id']),['sector_id','configuration_id']])

    add_configuration_has_airspace(engine,data_configurations.loc[pd.isnull(data_configurations.loc[:,'sector_id']),['airspace_id','configuration_id']])

def file2db_openingschemes(engine,airac,fpath='.',filename='OpeningScheme_'):
    if filename.endswith('.cos'):
        filename=fpath+'/'+filename
        skiprows=1
    else:
        filename=fpath+'/'+filename+str(airac)+'.cos'
        skiprows=0

    conf_dict=read_configurations_ids(engine,airac)
    
    data_os = pd.read_csv(filename, sep=";", skiprows=skiprows, header=None)
    data_os.columns = ['date', 'acc_sid','start','end','conf','info_origin']

    data_os['date']=data_os['date'].apply(format_date)
    data_os['start']=data_os.apply(lambda x: x['date']+" "+x['start']+":00", axis=1)
    data_os['end']=data_os.apply(lambda x: x['date']+" "+x['end']+":00", axis=1)
    data_os['configuration_id']=data_os.apply(lambda x: conf_dict.get(x['acc_sid']+"_"+x['conf']), axis=1)
    data_os['airac']=airac

    add_opening_scheme(engine,data_os)

def file2db_flow(engine,airac,fpath='.',filename='Flow_'):
    if filename.endswith('.nflw'):
        filename=fpath+'/'+filename
        skiprows=0
    else:
        filename=fpath+'/'+filename+str(airac)+'.nflw'
        skiprows=0

    data = pd.read_csv(filename, sep=";", skiprows=1, header=None, names=["FE","sid","name_type","sense"])
    data['airac']=airac
    df=data.loc[data['FE']=="F"]
    df.columns=['F','sid','name','ne','airac']

    add_flow(engine,df)
    f_dict=read_flow_ids(engine,airac)

    data['flow_id']=data.apply(lambda x: f_dict.get(x['sid'],np.nan) if x['FE']=="F" else np.nan, axis=1)

    data['flow_id']=data['flow_id'].fillna(method='pad')

    sect_dict=read_sectors_ids(engine,airac)
    air_dict=read_airspace_ids(engine,airac)
    wpt_dict=read_waypoints_ids(engine)
    airp_dict=read_airports_id(engine)
    airset_dict=read_airport_sets_ids(engine)


    data['sector_id']=data['sid'].apply(lambda x: sect_dict.get(x))
    data['airspace_id']=data['sid'].apply(lambda x: air_dict.get(x))
    data['waypoint_id']=data['sid'].apply(lambda x: wpt_dict.get(x))
    data['airport_id']=data['sid'].apply(lambda x: airp_dict.get(x))
    data['airportset_id']=data['sid'].apply(lambda x: airset_dict.get(x))


    add_flow_has_airportset(engine,data.loc[data["name_type"]=="AZ",['flow_id','airportset_id','sense']])
    add_flow_has_airport(engine,data.loc[data["name_type"]=="AD",['flow_id','airport_id','sense']])
    add_flow_has_waypoint(engine,data.loc[data["name_type"]=="SP",['flow_id','waypoint_id','sense']])
    add_flow_has_sector(engine,data.loc[(data["name_type"]=="AS") & (~pd.isnull(data["sector_id"])) ,['flow_id','sector_id','sense']])
    add_flow_has_airspace(engine,data.loc[(data["name_type"]=="AS") & (~pd.isnull(data["airspace_id"])) ,['flow_id','airspace_id','sense']])

def file2db_trafficvolume(engine,airac,fpath='.',filename='TrafficVolume_'):
    if filename.endswith('.ntfv'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.ntfv'
    
    data = pd.read_csv(filename, sep=";", skiprows=1, header=None,index_col = False,\
                       names=["TF","sid","1","2","3","4","5","6","7","8"])

    flow_dict=read_flow_ids(engine,airac)

    data['flow_id']=data.apply(lambda x: flow_dict.get(x['sid']) if x['TF']=='F' else np.NaN, axis=1)

    data.columns=['TF','sid','name','category','loc','loctype','sense','min_fl','max_fl','8','flow_id']
    data['airac']=airac

    add_trafficvolume(engine,data.loc[(data["TF"]=="T") ,['airac','sid','name','category']])

    tv_dict=read_trafficvolume_ids(engine,airac)

    data['trafficvolume_id']=data.apply(lambda x: tv_dict.get(x['sid']) if x["TF"]=="T" else np.nan, axis=1)

    data['trafficvolume_id']=data['trafficvolume_id'].fillna(method='pad')

    data_t=data.loc[(data["TF"]=="T") ,['TF','sid','name','category','loc','loctype','sense','min_fl','max_fl','8','flow_id',"trafficvolume_id"]]
    data_f=data.loc[(data["TF"]=="F") ,["sid","name","flow_id","trafficvolume_id"]]
    data_f.columns=["sid","type","flow_id","trafficvolume_id"]

    sect_dict=read_sectors_ids(engine,airac)
    air_dict=read_airspace_ids(engine,airac)
    wpt_dict=read_waypoints_ids(engine)
    airp_dict=read_airports_id(engine)
    airset_dict=read_airport_sets_ids(engine)

    data_t['sector_id']=data_t['loc'].apply(lambda x: sect_dict.get(x))
    data_t['airspace_id']=data_t['loc'].apply(lambda x: air_dict.get(x))
    data_t['waypoint_id']=data_t['loc'].apply(lambda x: wpt_dict.get(x))
    data_t['airport_id']=data_t['loc'].apply(lambda x: airp_dict.get(x))
    data_t['airportset_id']=data_t['loc'].apply(lambda x: airset_dict.get(x))

    add_trafficvolume_has_flow(engine,data_f)
    add_trafficvolume_has_sector(engine,data_t.loc[(data_t["loctype"]=="AS") & (~pd.isnull(data_t["sector_id"])),['trafficvolume_id','sector_id']])
    add_trafficvolume_has_airspace(engine,data_t.loc[(data_t["loctype"]=="AS") & (~pd.isnull(data_t["airspace_id"])),['trafficvolume_id','airspace_id']])
    add_trafficvolume_has_airportset(engine,data_t.loc[(data_t["loctype"]=="AZ") & (~pd.isnull(data_t["airportset_id"])),['trafficvolume_id','airportset_id','sense']])
    add_trafficvolume_has_airport(engine,data_t.loc[(data_t["loctype"]=="AD") & (~pd.isnull(data_t["airport_id"])),['trafficvolume_id','airport_id','sense']])
    add_trafficvolume_has_waypoint(engine,data_t.loc[(data_t["loctype"]=="SP") & (~pd.isnull(data_t["waypoint_id"])),['trafficvolume_id','waypoint_id','min_fl','max_fl','sense']])

def file2db_regulation(engine,airac,fpath='.',filename='RegPlan_'):
    if filename.endswith('.nreg'):
        filename=fpath+'/'+filename
    else:
        filename=fpath+'/'+filename+str(airac)+'.nreg'
    
    data = pd.read_csv(filename,sep=" ",skiprows=0, header=None)

    if len(data.columns)==11:
        #Old format
        data.columns=["sid","loc","loc_type","sdate","shour","edate","ehour",\
                             "slotwindow_width","slotslice_width","reason","nperiods"]
        
        data['key']=data['sid'].apply(lambda x: x if ":" not in x else None)
        data['key']=data['key'].fillna(method='ffill')
        data['period'] = data.groupby(['key']).cumcount()+1
        data['period'] = data['period']-1
        
        
    elif len(data.columns)==9:
        #New format
        data.columns=["sdate","sid","loc","loc_type","prev_day","slotwindow_width","slotslice_width","reason","nperiods"]
        dict_next_days=data.loc[data['prev_day']=="Y",['sid','sdate']].set_index('sid')['sdate'].to_dict()
        data['edate']=data['sid'].apply(lambda x: dict_next_days.get(x,None))
        data.loc[data['edate'].isnull(),'edate']=data.loc[data['edate'].isnull(),'sdate']  
        
        min_date=min(data.loc[data['reason'].notnull(),'sdate'])
        data.loc[(data['sdate']==min_date) & (data['prev_day']=="Y"),'sdate']=\
            data.loc[(data['sdate']==min_date) & (data['prev_day']=="Y"),'sdate'].apply(lambda x: (datetime.strptime(x,'%Y%m%d')\
                                                                                        - timedelta(days=1)).strftime("%Y%m%d"))

        min_date=min(data.loc[data['reason'].notnull(),'sdate'])
        data.loc[(data['sdate']==min_date) & (data['prev_day']=="Y"),'prev_day']="N"
        
        data['prev_day']=data['prev_day'].fillna(method='ffill')
        data = data[data['prev_day']=="N"]
        
        data.loc[data['loc_type'].isnull(),'shour']=data.loc[data['loc_type'].isnull(),'sdate']
        data.loc[data['loc_type'].isnull(),'ehour']=data.loc[data['loc_type'].isnull(),'sid']
        
        data=data.drop({'prev_day'},axis=1)
        
        data['key']=data['sid'].apply(lambda x: x if ":" not in x else None)
        data['key']=data['key'].fillna(method='ffill')
        data['nperiods']=data['nperiods'].fillna(method='ffill')
        
        data['period'] = data.groupby(['key']).cumcount()+1
        data['period'] = data['period']-1
        dict_ehour=data.loc[data['nperiods']==data['period'],['key','sid']].set_index('key')['sid'].to_dict()
        dict_shour=data.loc[1==data['period'],['key','sdate']].set_index('key')['sdate'].to_dict()
        
        data['shour']=data['sid'].apply(lambda x: dict_shour.get(x,x))
        data['ehour']=data['sid'].apply(lambda x: dict_ehour.get(x,x))
        
        data.loc[data['slotwindow_width'].isnull(),'loc_type']=data.loc[data['slotwindow_width'].isnull(),'loc']
        data.loc[data['slotwindow_width'].isnull(),'loc']=data.loc[data['slotwindow_width'].isnull(),'sid']
        data.loc[data['slotwindow_width'].isnull(),'sid']=data.loc[data['slotwindow_width'].isnull(),'sdate']
        data.loc[data['slotwindow_width'].isnull(),'sdate']=None
        data.loc[data['slotwindow_width'].isnull(),'shour']=None
        data.loc[data['slotwindow_width'].isnull(),'edate']=None
        data.loc[data['slotwindow_width'].isnull(),'ehour']=None
        data.loc[data['slotwindow_width'].isnull(),'nperiods']=None
        

    dict_first_hstart = data.loc[data['period']==1,['key','sid']].set_index('key')['sid'].to_dict()
    dict_sdate = data[['sid','sdate']].set_index('sid')['sdate'].to_dict()
    dict_edate = data[['sid','edate']].set_index('sid')['edate'].to_dict()

        

    data.loc[data['sdate'].isnull(),'sdate']=data.loc[data['sdate'].isnull()].apply(lambda x:
                                                                                           dict_sdate.get(x['key']) 
                                                                                            if (x['sid']>=dict_first_hstart.get(x['key']))
                                                                                            else dict_edate.get(x['key']), axis=1)

    data.loc[data['edate'].isnull(),'edate']=data.loc[data['edate'].isnull()].apply(lambda x:
                                                                                           dict_sdate.get(x['key']) 
                                                                                            if (x['loc']>=dict_first_hstart.get(x['key']))
                                                                                            else dict_edate.get(x['key']), axis=1)

        
    data = data[["sid","loc","loc_type","sdate","shour","edate","ehour",\
                             "slotwindow_width","slotslice_width","reason","nperiods","period","key"]].reset_index(drop=True)

    data['airac']=airac

    data["start"]=data.apply(lambda x: np.nan if pd.isnull(x["sdate"]) else str(x["sdate"])[0:4]+"-"+str(x["sdate"])[4:6]+"-"+
                             str(x["sdate"])[6:8]+" "+str(x["shour"])+":00", axis=1)
    data["end"]=data.apply(lambda x: np.nan if pd.isnull(x["edate"]) else str(x["edate"])[0:4]+"-"+str(x["edate"])[4:6]+"-"+
                             str(x["edate"])[6:8]+" "+str(x["ehour"])+":00", axis=1)


    tv_dict=read_trafficvolume_ids(engine,airac)
    airs_dict=read_airspace_ids(engine,airac)

    data["trafficvolume_id"]=data.apply(lambda x: tv_dict.get(x["loc"]) if x["loc_type"]=="TV" else np.nan, axis=1)
    data["airspace_id"]=data.apply(lambda x: airs_dict.get(x["loc"]) if x["loc_type"]=="AS" else np.nan, axis=1)

    add_regulation(engine,data.loc[(~pd.isnull(data.loc[:,"reason"])),["airac","sid","trafficvolume_id","airspace_id","start","end","slotwindow_width","slotslice_width","reason"]])


    reg_dict=read_regulation(engine,airac)
    data["regulation_id"]=data["sid"].apply(lambda x: reg_dict.get(x,np.nan))

    data = data.fillna(method='ffill')

    data["RP"]=data["sid"].apply(lambda x: reg_dict.get(x,"P"))
    data=data.loc[(data.loc[:,"RP"]=="P"),["sid","loc","loc_type","regulation_id","sdate","edate"]]
    data.columns=["hs","he","capacity","regulation_id","sdate","edate"]

    data["start"]=data.apply(lambda x: str(x["sdate"])[0:4]+"-"+str(x["sdate"])[4:6]+"-"+
                             str(x["sdate"])[6:8]+" "+x["hs"]+":00", axis=1)
    data["end"]=data.apply(lambda x: str(x["edate"])[0:4]+"-"+str(x["edate"])[4:6]+"-"+
                             str(x["edate"])[6:8]+" "+str(x["he"])+":00", axis=1)
    
    add_regulation_period(engine,data[["regulation_id","start","end","capacity"]])

def file2db_capacities(engine,airac,fpath='.',filename='Capacity_'):
    if filename.endswith('.ncap'):
        filename=fpath+'/'+filename
        skiprows=1
    else:
        filename=fpath+'/'+filename+str(airac)+'.ncap'
        skiprows=0

    data = pd.read_csv(filename,sep=";",skiprows=skiprows, header=None)
    data.columns=['date','element_id','start_time','end_time','capacity','hourly3capacity','element_type','element_cat','env_table_source']

    data['date']=data['date'].apply(lambda x: str(x)[6:]+"-"+str(x)[3:5]+"-"+str(x)[0:2])
    data['start']=data.apply(lambda x: x['date']+" "+x['start_time'], axis=1)
    data['end']=data.apply(lambda x: x['date']+" "+x['end_time'], axis=1)
    data['hourly3capacity']=data['hourly3capacity'].apply(lambda x: np.nan if str(x)=='_' else x)

    airsp_dict=read_airspace_ids(engine,airac)
    sect_dict=read_sectors_ids(engine,airac)
    airports_dict=read_airports_id(engine)
    airport_set_dict = read_airport_sets_ids(engine)
    tv_dict = read_trafficvolume_ids(engine,airac)
    wpt_dict=read_waypoints_ids(engine)

    data_as = data.loc[data['element_type']=="AS"].copy()
    data_as['sector_id']=data_as['element_id'].apply(lambda x: sect_dict.get(x,None))
    
    data_aa = data.loc[data['element_type']=="AS"].copy()
    data_aa['airspace_id']=data_aa['element_id'].apply(lambda x: airsp_dict.get(x,None))

    data_ad = data.loc[data['element_type']=="AD"].copy()
    data_ad['airport_id']=data_ad['element_id'].apply(lambda x: airports_dict.get(x,None))
    
    data_az = data.loc[data['element_type']=="AZ"].copy()
    data_az['airportset_id']=data_az['element_id'].apply(lambda x: airport_set_dict.get(x,None))

    data_tv = data.loc[data['element_type']=="TV"].copy()
    data_tv['trafficvolume_id']=data_tv['element_id'].apply(lambda x: tv_dict.get(x,None))

    data_pt = data.loc[data['element_type']=="PT"].copy()
    data_pt['waypoint_id']=data_pt['element_id'].apply(lambda x: wpt_dict.get(x,None))

    add_capacities(engine,data_as.loc[data_as['sector_id'].notnull()].copy(),'capacity_sector','sector_id')
    add_capacities(engine,data_aa.loc[data_aa['airspace_id'].notnull()].copy(),'capacity_airspace','airspace_id')
    add_capacities(engine,data_ad.loc[data_ad['airport_id'].notnull()].copy(),'capacity_airport','airport_id')
    add_capacities(engine,data_az.loc[data_az['airportset_id'].notnull()].copy(),'capacity_airportset','airportset_id')
    add_capacities(engine,data_tv.loc[data_tv['trafficvolume_id'].notnull()].copy(),'capacity_trafficvolume','trafficvolume_id')
    add_capacities(engine,data_pt.loc[data_pt['waypoint_id'].notnull()].copy(),'capacity_waypoint','waypoint_id')

def add_capacities(engine,data,table,element_id):
    if len(data)>0:
        data.rename(columns={'element_id': element_id,'element_cat':'category'}, inplace=True)
        data[[element_id,'start','end','capacity','hourly3capacity','category','env_table_source']].to_sql(table,engine,index=False,if_exists="append")

def add_airports(engine,airports_icao):
    d=pd.DataFrame(list(airports_icao))
    d.columns=['icao_id']
    d[['icao_id']].to_sql("airport",engine, index=False, if_exists="append")

def add_airport_point_connection(engine,pd):
    pd[['airport_id','geopoint_id','airac','name','altitude','tis',
      'trs','taxi_time']].to_sql("airport_has_geopoint",engine, index=False, if_exists="append")

def add_waypoint(engine,pd):
    pd[['airac','sid','type']].to_sql("waypoint",engine, index=False, if_exists="append")

def add_waypoint_point_connection(engine,pd):
     pd[['waypoint_id','geopoint_id','airac',
         'name']].to_sql("waypoint_has_geopoint",engine, index=False, if_exists="append")

def add_airportsets(engine,pd):
     pd[['sid']].to_sql("airportset",engine, index=False, if_exists="append")    

def add_airportset_has_airport(engine,pd):
    pd[['airportset_id','airport_id','airac','name','type']].to_sql("airportset_has_airport",engine, index=False, if_exists="append") 

def add_airblocks(engine,pd):    
    pd[['name', 'airac', 'poly']].to_sql("temp_poly",engine, index=False, if_exists="replace")

    engine.execute("insert into airblock(airac,name,boundary) "+
                 "select airac, name, ST_GeomFromText(poly) from temp_poly")
    
    engine.execute("drop table temp_poly")
    
def add_sectors(engine,pd):
     pd[['airac','sid','name','type','category']].to_sql("sector",engine, index=False, if_exists="append")    

def add_sectorslice(engine,pd):
    pd[['sector_id','airblock_id','lower_fl','upper_fl','operation']].to_sql("sectorslice",engine, index=False, if_exists="append")

def add_airspace(engine,pd):
    pd[['airac','sid','type','name']].to_sql("airspace",engine, index=False, if_exists="append")

def add_airspace_has_airspace(engine,pd):
    pd[['parent_airspace_id','subairspace_id']].to_sql("airspace_has_airspace",engine, index=False, if_exists="append")
    
def add_airspace_has_sector(engine,pd):
    pd[['airspace_id','sector_id']].to_sql("airspace_has_sector",engine, index=False, if_exists="append")
    
def add_controllers_availability(engine,pd):
    pd[['airac','airspace_id','start','end','max_controllers_pos']].to_sql("controllers_availability",engine, index=False, if_exists="append")

def add_configuration(engine,pd):
    pd[['sid','num_controllers','acc_id','airac']].to_sql("configuration",engine, index=False, if_exists="append")

def add_configuration_has_sector(engine,pd):
    pd[['sector_id','configuration_id']].to_sql("configuration_has_sector",engine, index=False, if_exists="append")

def add_configuration_has_airspace(engine,pd):
    pd[['airspace_id','configuration_id']].to_sql("configuration_has_airspace",engine, index=False, if_exists="append")

def add_opening_scheme(engine,pd):
    pd[['airac','configuration_id','start','end','info_origin']].to_sql("opening_scheme",engine, index=False, if_exists="append")

def add_flow(engine,pd):
    pd[['airac','sid','name']].to_sql("flow",engine, index=False, if_exists="append")

def add_flow_has_airportset(engine,pd):
    pd[['flow_id','airportset_id','sense']].to_sql("flow_has_airportset",engine, index=False, if_exists="append")
    
def add_flow_has_airport(engine,pd):
    pd[['flow_id','airport_id','sense']].to_sql("flow_has_airport",engine, index=False, if_exists="append")

def add_flow_has_waypoint(engine,pd):
    pd[['flow_id','waypoint_id','sense']].to_sql("flow_has_waypoint",engine, index=False, if_exists="append")

def add_flow_has_sector(engine,pd):
    pd[['flow_id','sector_id','sense']].to_sql("flow_has_sector",engine, index=False, if_exists="append")

def add_flow_has_airspace(engine,pd):
    pd[['flow_id','airspace_id','sense']].to_sql("flow_has_airspace",engine, index=False, if_exists="append")
    
def add_trafficvolume(engine,pd):
    pd[['airac','sid','name','category']].to_sql("trafficvolume",engine, index=False, if_exists="append")

def add_trafficvolume_has_flow(engine,pd):
    pd[['trafficvolume_id','flow_id','type']].to_sql("trafficvolume_has_flow",engine, index=False, if_exists="append")
    
def add_trafficvolume_has_sector(engine,pd):
    pd[['trafficvolume_id','sector_id']].to_sql("trafficvolume_has_sector",engine, index=False, if_exists="append")

def add_trafficvolume_has_airspace(engine,pd):
    pd[['trafficvolume_id','airspace_id']].to_sql("trafficvolume_has_airspace",engine, index=False, if_exists="append")

def add_trafficvolume_has_airport(engine,pd):
    pd[['trafficvolume_id','airport_id','sense']].to_sql("trafficvolume_has_airport",engine, index=False, if_exists="append")

def add_trafficvolume_has_airportset(engine,pd):
    pd[['trafficvolume_id','airportset_id','sense']].to_sql("trafficvolume_has_airportset",engine, index=False, if_exists="append")

def add_trafficvolume_has_waypoint(engine,pd):
    pd[['trafficvolume_id','waypoint_id','min_fl','max_fl','sense']].to_sql("trafficvolume_has_waypoint",engine, index=False, if_exists="append")
    
def add_regulation(engine,pd):
    pd[["airac","sid","trafficvolume_id","airspace_id","start","end","slotwindow_width","slotslice_width","reason"]].to_sql("regulation",engine, index=False, if_exists="append")

def add_regulation_period(engine,pd):
    pd[["regulation_id","start","end","capacity"]].to_sql("regulation_period",engine, index=False, if_exists="append")

def add_ddr_to_ddr_databases(engine,airac,database,deleteAIRAC=False):
    if deleteAIRAC:
        engine.execute("DELETE d FROM ddr_databases d WHERE d.airac=\""+str(airac)+"\"")
    pd.DataFrame([{'airac':airac, 'ddr_name':database, 'ddr_2_days':0, 'ddr_3_days':0, 'ddr_4_days':0, 'ddr_individual_days':0}]).to_sql("ddr_databases", engine,  index=False, if_exists="append")

def read_geopoints_ids(engine,from_p_id=-1,rounded=0,type_point=None):   
        geopt_dict={}


        if type_point is None:
            cur = engine.execute("SELECT p.id, p.type, ST_X(p.coords), ST_Y(p.coords) "
                       "FROM geopoint p WHERE p.id> "+str(from_p_id))
        else:
            cur = engine.execute("SELECT p.id, p.type, ST_X(p.coords), ST_Y(p.coords) "
                       "FROM geopoint p WHERE p.type=\""+t+"\" AND p.id> "+str(from_p_id))
        
        for row in cur:
            if rounded==1:
                geopt_dict[str(truncate(row[2], 12))+"_"+str(truncate(row[3], 12))]={'id':row[0], 'type':row[1], 'lat':row[2], 'lon':row[3]};
            else:
                geopt_dict[str(row[2])+"_"+str(row[3])]={'id':row[0], 'type':row[1], 'lat':row[2], 'lon':row[3]};
        
        return geopt_dict

def add_geopoints(engine, d): 
  d['geomText']=d['latlon'].apply(lambda x: "POINT( "+x.replace("_"," ")+" )")

  d[['geomText', 'type']].to_sql("temp_point_test",engine, index=False, if_exists="replace")

  engine.execute("insert into geopoint(coords,type) "+
               "select ST_GeomFromText(geomText), type from temp_point_test")

  engine.execute("drop table temp_point_test")

def read_airports_id(engine, airac=None):
  #Read airports ids that have a geo_point defined for a given airac
  #if airac not given, i.e., airac=None then load all airports
  sql = "SELECT a.id, a.icao_id FROM airport a"

  if airac is not None:
      sql = sql + "JOIN airport_has_geopoint ahg ON ahg.airport_id=a.id \
                  WHERE ahg.airac="+str(airac)
  
  df = pd.read_sql(sql, engine)
  airport_dict = pd.Series(df.id.values,index=df.icao_id).to_dict()
  return airport_dict

def read_airport_sets_ids(engine):
    sql = "SELECT a_s.id, a_s.sid FROM airportset a_s"
    df = pd.read_sql(sql, engine)
    as_dict = pd.Series(df.id.values,index=df.sid).to_dict()
    return as_dict

def read_waypoints_ids(engine):
    sql = "SELECT w.id, w.sid FROM waypoint w"
    df = pd.read_sql(sql, engine)
    wpt_dict=pd.Series(df.id.values,index=df.sid).to_dict()
    return wpt_dict

def read_airblocks_ids(engine,airac):
        sql="SELECT ab.id, ab.name FROM airblock ab WHERE airac="+str(airac)
        df = pd.read_sql(sql, engine)
        ab_dict=pd.Series(df.id.values,index=df.name).to_dict()
        return ab_dict

def read_sectors_ids(engine,airac):
        sql = "SELECT s.id, s.sid  FROM sector s WHERE airac="+str(airac)
        df = pd.read_sql(sql, engine)
        sect_dict = pd.Series(df.id.values,index=df.sid).to_dict()
        return sect_dict

def read_airspace_ids(engine,airac):
        sql = "SELECT a.id, a.sid  FROM airspace a WHERE airac="+str(airac)
        df = pd.read_sql(sql, engine)
        as_dict = pd.Series(df.id.values,index=df.sid).to_dict()
        return as_dict

def read_configurations_ids(engine,airac):
    conf_dict={}
    cur=engine.execute("SELECT c.id, c.sid, a.sid FROM configuration c JOIN airspace a on c.acc_id=a.id WHERE c.airac="+str(airac))
    for row in cur:
        conf_dict[row[2]+"_"+row[1]]=row[0];
            
    return conf_dict

def read_flow_ids(engine,airac):
    sql = "SELECT f.id, f.sid FROM flow f WHERE airac="+str(airac)
    df = pd.read_sql(sql, engine)
    f_dict = pd.Series(df.id.values,index=df.sid).to_dict()
    return f_dict

def read_trafficvolume_ids(engine,airac):
    sql="SELECT t.id, t.sid FROM trafficvolume t WHERE airac="+str(airac)
    df = pd.read_sql(sql, engine)
    tv_dict = pd.Series(df.id.values,index=df.sid).to_dict()
    return tv_dict

def read_regulation(engine,airac):
    sql="SELECT r.id, r.sid FROM regulation r WHERE airac="+str(airac)
    df = pd.read_sql(sql, engine)
    reg_dict = pd.Series(df.id.values,index=df.sid).to_dict()
    return reg_dict

def read_waypoints_airac_with_geopoints(engine, airac):
    sql="SELECT w.id as waypoint_id, w.sid as sid, g.id as geo_id, \
          SUBSTRING_INDEX(SUBSTR(ST_ASTEXT(g.coords),7),')',1) as coords FROM waypoint w \
          JOIN waypoint_has_geopoint whg ON whg.waypoint_id=w.id \
          JOIN geopoint g on g.id=whg.geopoint_id \
          WHERE whg.airac="+str(airac)

    df = pd.read_sql(sql, engine)

    df['coords']=df['coords'].str.decode('utf-8')

    wpt_geo_dict=df.set_index('sid')[['waypoint_id','geo_id','coords']].to_dict('index')
    
    return wpt_geo_dict

def read_airport_with_geopoints(engine, airac, include_missing=False, excluding=[]):
    #Read airports with geopoint in an airac
    #If include_missing=True then add airports missing in that airac by looking for closest geopoint
    #excluding include airports where geopoint will be -1
    sql="SELECT a.id as airport_id, a.icao_id as icao_id, g.id as geo_id, \
        SUBSTRING_INDEX(SUBSTR(ST_ASTEXT(g.coords),7),')',1) as coords \
        FROM airport a \
        JOIN airport_has_geopoint ahg ON ahg.airport_id=a.id \
        JOIN geopoint g on g.id=ahg.geopoint_id \
        WHERE ahg.airac="+str(airac)

    df = pd.read_sql(sql, engine)
    df['coords']=df['coords'].str.decode('utf-8')

    as_geo_dict = df.set_index('icao_id')[['airport_id','geo_id','coords']].to_dict('index')

    if include_missing:

        #Read airports are missing in the airac
        sql = "SELECT missing_airports_in_airac.id as airport_id, missing_airports_in_airac.icao_id as icao_id, \
            ahg.geopoint_id as geo_id, SUBSTRING_INDEX(SUBSTR(ST_ASTEXT(g.coords),7),')',1) AS coords, \
            (ABS(ahg.airac-"+str(airac)+")) AS diff_airac \
            FROM \
            (SELECT a.* \
            FROM airport a \
            LEFT JOIN \
            (SELECT a.id FROM airport a \
            LEFT JOIN airport_has_geopoint ahg ON ahg.airport_id=a.id \
            where ahg.airac="+str(airac)+") AS a_airac \
            ON a.id=a_airac.id where a_airac.id is null) AS missing_airports_in_airac \
            join airport_has_geopoint ahg ON ahg.airport_id=missing_airports_in_airac.id \
            join geopoint g ON ahg.geopoint_id=g.id"

        df = pd.read_sql(sql, engine)
        df['coords']=df['coords'].str.decode('utf-8')

        as_geo_dict.update(df.set_index('icao_id')[['airport_id','geo_id','coords']].to_dict('index'))

        
        
        if len(excluding)>0: 

            #Select airports that are missing and check if are one of excluded
            sql="SELECT a.id as airport_id, a.icao_id as icao_id, -1 as geo_id, \"Nan Nan\" as coords \
                FROM airport a \
                LEFT JOIN \
                (SELECT a.id FROM airport a \
                LEFT JOIN airport_has_geopoint ahg ON ahg.airport_id=a.id \
                WHERE ahg.airac="+str(airac)+") AS a_airac \
                ON a.id=a_airac.id WHERE a_airac.id IS NULL AND a.icao_id IN ("+str(excluding).replace('[','').replace(']','')+")"

            df = pd.read_sql(sql, engine)

            as_geo_dict.update(df.set_index('icao_id')[['airport_id','geo_id','coords']].to_dict('index'))


    return as_geo_dict

def read_maxid_coordpoints(engine):
    sql="SELECT max(c.id) as max_id FROM coordpoint c"
    df = pd.read_sql(sql, engine)
    max_id=df.loc[0,'max_id']
    if max_id is None:
        max_id=0
    return max_id

#READ AIRSPACE DATA
def read_airspace_information(engine,airac, include_missing=False, excluding=["ZZZZ", "AFIL"], include_waypoints=True):
    #READ DICTIONARIES OF AIRSPACE INFORMATION -- I have inreased the buffer in the my. msql to speed up the fetching of the results
    regs_dict=read_regulation(engine,airac)
    wpt_geo_dict=read_waypoints_airac_with_geopoints(engine,airac)
    airports_geo_dict=read_airport_with_geopoints(engine,airac,include_missing, excluding)
    airsp_dict=read_airspace_ids(engine,airac)
    sect_dict=read_sectors_ids(engine,airac)
    airports_dict=read_airports_id(engine)
    coord_geo_dict = {}
    if include_waypoints:
      coord_geo_dict=read_coordpoints_with_geopoints(engine) #caution that coord_geo_dict might change as we add points **
    max_coord_id=read_maxid_coordpoints(engine)
    
    return regs_dict, wpt_geo_dict, airports_geo_dict, airsp_dict, sect_dict, airports_dict, coord_geo_dict, max_coord_id






##ALL_FT+ FLIGHT DATA

def add_flight_details_in_db(engine, data, airac,verbose=False, airports_geo_dict=None,airports_dict=None,wpt_geo_dict=None,
                                airsp_dict=None,sect_dict=None,coord_geo_dict=None,max_coord_id=None,excluding=["ZZZZ","AFIL"], 
                                ddr_version=None, add_flights=True, add_additional_info=True, add_trajectories=True, add_airspace=True,
                                add_circles=True,add_request=True, add_estimatedtime=True, if_conflict="ignore", fids_dict=None,
                                load_ftfm=True, load_rtfm=True, load_ctfm=True, load_scr=True,
                                load_srr=True, load_sur=True, load_dct=True, load_cpf=True):
        
        start = time.time()

        if verbose:
            print("Number flights data orig: "+str(len(data)))

        data = process_conflict_flights(engine,data,airac,if_conflict,fids_dict=fids_dict,verbose=verbose)

        if verbose:
            print("Number flights data after removing duplicates: "+str(len(data)))

        if len(data)==0:
            #There are no flights to be processed
            return coord_geo_dict, max_coord_id, fids_dict

        set_foreing_key_checks(engine,0)

        if add_flights:
            t0=time.time()
            add_flight(engine,data)
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FLIGHT in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))

        #READ FLIGHT ID FROM DB TO ADD DB_ID INTO PANDA DATA STRUCTURE
        fids_dict=read_flight_id(engine,airac)
        t0=time.time()
        data['flight_id']=data.apply(lambda x: fids_dict.get(x['ifps_id'])
                                     if not pd.isnull(x['ifps_id']) \
                                     else fids_dict.get(x['tact_id']), axis=1) 

        dict_ftfm=read_trajectory(engine,"ftfm",flight_id_list=list(data['flight_id']))

        t1=time.time()
        if verbose:
            t=t1-t0
            print("ADDED FLIGHT IDs in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))

        if add_additional_info:
            t0=time.time()
            #ADDITIONAL_FLIGHT_INFO
            add_atfm_info(engine,data)
            add_cdm_info(engine,data)
            add_intention_info(engine,data)
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FLIGHT INFO in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))

        if add_trajectories:
            t0=time.time()
            coord_geo_dict, max_coord_id, dict_ftfm, dict_rtfm, dict_ctfm, \
            dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf=\
            add_flight_trajectories(engine,data,
                                    airports_geo_dict, wpt_geo_dict, airsp_dict,
                                    sect_dict, coord_geo_dict,
                                    max_coord_id,ddr_version,excluding,verbose,
                                    load_ftfm=load_ftfm, load_rtfm=load_rtfm, load_ctfm=load_ctfm, load_scr=load_scr,
                                    load_srr=load_srr, load_sur=load_sur, load_dct=load_dct, load_cpf=load_cpf) ##excluding are airport which latlon will be added as a geopoint and not rely on the coordinates of the airport. Namely ZZZZ and AFIL
            
            
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FLIGHT TRAJECTORIES in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60))))) #2.5 h

        else:
            dict_ftfm=read_trajectory(engine,"ftfm",flight_id_list=list(data['flight_id']))
            dict_rtfm=read_trajectory(engine,"rtfm",flight_id_list=list(data['flight_id']))
            dict_ctfm=read_trajectory(engine,"ctfm",flight_id_list=list(data['flight_id']))
            dict_scr=read_trajectory(engine,"scr",flight_id_list=list(data['flight_id']))
            dict_srr=read_trajectory(engine,"srr",flight_id_list=list(data['flight_id']))
            dict_sur=read_trajectory(engine,"sur",flight_id_list=list(data['flight_id']))
            dict_dct=read_trajectory(engine,"dct",flight_id_list=list(data['flight_id']))
            dict_cpf=read_trajectory(engine,"cpf",flight_id_list=list(data['flight_id']))
            
            
        if add_airspace:
            t0=time.time()
            coord_geo_dict, max_coord_id = add_flight_airspaces(engine,data, airsp_dict, sect_dict, coord_geo_dict, 
                                                dict_ftfm, dict_rtfm, dict_ctfm, dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf,
                                                max_coord_id, ddr_version,verbose,
                                                load_ftfm=load_ftfm, load_rtfm=load_rtfm, load_ctfm=load_ctfm, load_scr=load_scr,
                                                load_srr=load_srr, load_sur=load_sur, load_dct=load_dct, load_cpf=load_cpf)
            
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FLIGHT AIRSPACES in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60))))) #6 h

            
        if add_circles:
            t0=time.time()
            coord_geo_dict, max_coord_id=add_flight_circles_intersections(engine,data, coord_geo_dict, 
                                                            dict_ftfm, dict_rtfm, dict_ctfm, dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf,
                                                            airports_dict,
                                                            max_coord_id,ddr_version,verbose,
                                                            load_ftfm=load_ftfm, load_rtfm=load_rtfm, load_ctfm=load_ctfm, load_scr=load_scr,
                                                            load_srr=load_srr, load_sur=load_sur, load_dct=load_dct, load_cpf=load_cpf)
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FLIGHT CIRCLES in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))


        if add_request:
            t0=time.time()
            add_flight_fl_requests(engine,data, dict_ftfm, dict_rtfm, dict_ctfm,verbose,load_ftfm=load_ftfm, load_rtfm=load_rtfm, load_ctfm=load_ctfm)
            t1=time.time()
            if verbose:
                t=t1-t0
                print("ADDED FL REQUESTS in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))   


        if add_estimatedtime:
            if load_ftfm:
                t0=time.time()
                coord_geo_dict,max_coord_id=add_eets(engine,data,sect_dict,wpt_geo_dict,airports_geo_dict,coord_geo_dict,dict_ftfm,max_coord_id,verbose)
                t1=time.time()
                if verbose:
                    t=t1-t0
                    print("ADDED FLIGHT EETS in: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))

        
        set_foreing_key_checks(engine,1)

        if verbose:
            t=time.time()-start
            print("ALL FT+ ADDED IN TOTAL OF: "+str(int(t/60/60))+":"+str(int((t-60*60*(int(t/3600)))/60))+":"+str(round(t-60*(int(t/60)))))

        return coord_geo_dict, max_coord_id, fids_dict

def process_conflict_flights(engine,data,airac,if_conflict="ignore",fids_dict=None,verbose=False,extra_condition=None):

        if fids_dict is None:
            if verbose:
                print("Loading flights dict in conflict")
            if extra_condition is None:
                fids_dict=read_flight_id(engine,airac)#,extra_condition="ddr_source not like \"%0901%\"") #HEREHERE
            else:
                fids_dict=read_flight_id(engine,airac,extra_condition=extra_condition)

        data['already_in_db']=data.apply(lambda x: fids_dict.get(x['ifps_id'])
                             if not pd.isnull(x['ifps_id']) \
                             else fids_dict.get(x['tact_id'],np.nan), axis=1) 

        if if_conflict == "ignore":
            #Remove from data the flights we already have. Copy only filghts where already_in_db is null
            if verbose:
                print("Ignoring: "+str(len(data[data['already_in_db'].notnull()])))

            data_r = data[data['already_in_db'].isnull()].copy().reset_index(drop=True, inplace=False)

        elif if_conflict == "replace":
            #Remove from database the information of the flights which have already_in_db. The points
            #and geopoitns will not be removed as they might be used by other flights and that might
            #make things complicated.
            if verbose:
                print("Replacing: "+str(len(data[data['already_in_db'].notnull()])))

            delete_flights_info_db(engine,data[data['already_in_db'].notnull() & data['ifps_id'].notnull()]['ifps_id'].tolist(),
                                        data[data['already_in_db'].notnull() & data['tact_id'].notnull() & data['ifps_id'].isnull()]['tact_id'].tolist())

            data['already_in_db']=np.nan
            data_r=data.copy()

        return data_r


def read_flight_id(engine,airac,extra_condition=None):
    sql="SELECT f.id as id, f.ifps_id as ifps_id, f.tact_id as tact_id FROM flight f \
         WHERE f.airac = "+str(airac)

    if extra_condition is not None:
        sql = sql + " AND "+extra_condition

    df = pd.read_sql(sql, engine)

    df.loc[df['ifps_id'].isnull(),'ifps_id']=df.loc[df['ifps_id'].isnull(),'tact_id']

    fids_dict = pd.Series(df.id.values,index=df.ifps_id).to_dict()  
    
    return fids_dict


def add_flight(engine,pd):    
    flight_database_fields=['airac', 'ifps_id', 'ac_id', 'tact_id', 'ac_type', 'ac_id_iata', 'airport_arrival', 'airport_departure',
'operator', 'aobt', 'iobt', 'cobt', 'eobt', 'lobt',  'original_flight_data_quality','flight_data_quality',
'source', 'late_filer', 'late_updater', 'north_atlantic_flight_status', 'flight_state', 'prev_to_activation_flight_state',
'sensitive_flight', 'operating_aircraft_operator_icao_id', 'runway_visual_range', 'arc_addr_source', 'arc_addr', 
'ifps_registration_mark', 'flight_type_icao', 'aircraft_equipment', 'no_cpgcpf_reason','ddr_version','ddr_source','individual_day']
                       
    pd[flight_database_fields].to_sql("flight",engine, index=False, if_exists="append")
       

def add_atfm_info(engine,data):
    flight_atfm_database_fields=['flight_id', 'excemption_reason_type', 'excemption_reason_distance', 'suspension_status', 
                             'sam_ctot', 'sam_sent', 'sip_ctot', 'sip_sent', 'slot_forced', 'most_penalising_reg',
                             'regulations_affected_by_nr_of_instances',
                             'reg_excluded_from_nr_of_instances', 'last_received_atfm_message_title',
                             'last_received_message_title', 'last_sent_atfm_message_title', 'manual_exemption_reason',
                             'ready_for_improvement', 'ready_to_depart', 'revised_taxi_time', 'tis', 'trs', 
                             'to_be_sent_slot_message_title', 'to_be_sent_proposal_message_title', 
                             'last_sent_slot_message_title', 'last_sent_proposal_message_title', 'last_sent_proposal_message',
                             'last_sent_slot_message', 'flight_count_option', 'normal_flight_tact_id', 'proposal_flight_tact_id',
                             'rerouting_why', 'rerouted_flight_state', 'number_ignored_errors']

    
    data[flight_atfm_database_fields].dropna(thresh=2).to_sql("flight_atfm_info",engine, index=False, if_exists="append")
    

    
def add_cdm_info(engine,data):
    flight_cdm_database_fields=['flight_id', 'cdm_status', 'cdm_early_ttot', 'cdm_ao_ttot', 'cdm_atc_ttot', 
                            'cdm_sequenced_ttot', 'cdm_taxi_time', 'cdm_off_block_time_discrepancy', 
                            'cdm_departure_procedure_id', 'cdm_aircraft_type_id', 'cdm_registration_mark', 
                            'cdm_no_slot_before', 'cdm_departure_status']

    data[flight_cdm_database_fields].dropna(thresh=2).to_sql("flight_cdm_info",engine, index=False, if_exists="append")


def add_intention_info(engine,data):
    flight_intention_database_fields=['flight_id', 'intention_flight', 'intention_related_route_assignment_method',
                                  'intention_uid', 'intention_edition_date', 'intention_source', 'associated_intetions',
                                  'enrichment_output']
                                             
    data[flight_intention_database_fields].dropna(thresh=2).to_sql("flight_intention_info",engine, index=False, if_exists="append")



def add_flight_trajectories(engine,data, airports_geo_dict, wpt_geo_dict, airsp_dict, sect_dict, coord_geo_dict, max_coord_id=-1,
                                ddr_version=3,excluded_airports=[""],verbose=False,
                                load_ftfm=True, load_rtfm=True, load_ctfm=True, load_scr=True,
                                load_srr=True, load_sur=True, load_dct=True, load_cpf=True):

    dict_ftfm={}
    dict_rtfm={}
    dict_ctfm={}
    dict_scr={}
    dict_srr={}
    dict_sur={}
    dict_dct={}
    dict_cpf={}
    
    if ddr_version==2:
        dict_aobt=pd.Series(data.aobt.values,index=data.flight_id).to_dict() #m3
        dict_cobt=pd.Series(data.cobt.values,index=data.flight_id).to_dict() #m2
        dict_lobt=pd.Series(data.lobt.values,index=data.flight_id).to_dict() #PRISME
        dict_eobt=pd.Series(data.eobt.values,index=data.flight_id).to_dict() #m1
    else:
        dict_aobt={}
        dict_cobt={}
        dict_lobt={}
        dict_eobt={}

    #READ POINT DATAFRAMES
    #Some points migth be missing, read the geopoints and if added then read again the dictionary after each execution
    #With DDR3 the dict_aob, dict_cobt and dict_eobt will not be used as date is in waypoint already

    if load_ftfm:
    
        if verbose:
            print("Points FTFM")
        
        ftfm_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'ftfmAllFtPointProfile', engine, 
                                                                        airports_geo_dict, wpt_geo_dict, coord_geo_dict, 
                                                                        max_coord_id,ddr_version,dict_eobt,excluded_airports)
        
        
        add_trajectory_ftfm(engine,data)
        dict_ftfm=read_trajectory(engine,"ftfm",flight_id_list=list(data['flight_id']))
        #The if not type == float is because if the flights do not have that trajectory, e.g., cpf, then dataframe_pointProfile returns nan instead of a frame and a nan is a float
        #therefore it is a way of checking if the trajectories exist
        if ftfm_point is not None:
            ftfm_point['trajectory_id']=ftfm_point['flight_id'].apply(lambda x: dict_ftfm.get(x))
            add_trajectory_points(engine,ftfm_point[['trajectory_id','geopoint_id','distance','fl','time_over','type',
                                                   'rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_rtfm:
        if verbose:
            print("Points RTFM")
        rtfm_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'rtfmAllFtPointProfile', engine,
                                                                        airports_geo_dict, wpt_geo_dict, coord_geo_dict,
                                                                        max_coord_id,ddr_version,dict_cobt,excluded_airports)
        add_trajectory_rtfm(engine,data)
        dict_rtfm=read_trajectory(engine,"rtfm",flight_id_list=list(data['flight_id']))
        if rtfm_point is not None:
            rtfm_point['trajectory_id']=rtfm_point['flight_id'].apply(lambda x: dict_rtfm.get(x))
            add_trajectory_points(engine,rtfm_point[['trajectory_id','geopoint_id','distance','fl','time_over','type',
                                              'rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_ctfm:
        if verbose:
            print("Points CTFM")    
        ctfm_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'ctfmAllFtPointProfile', engine,
                                                                        airports_geo_dict, wpt_geo_dict, coord_geo_dict,
                                                                        max_coord_id,ddr_version,dict_aobt,excluded_airports)
        add_trajectory_ctfm(engine,data)
        dict_ctfm=read_trajectory(engine,"ctfm",flight_id_list=list(data['flight_id']))
        if ctfm_point is not None:
            ctfm_point['trajectory_id']=ctfm_point['flight_id'].apply(lambda x: dict_ctfm.get(x))
            add_trajectory_points(engine,ctfm_point[['trajectory_id','geopoint_id','distance','fl','time_over','type',
                                              'rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_scr:
        if verbose:
            print("Points SCR")
        scr_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'scrAllFtPointProfile',engine, 
                                                                       airports_geo_dict, wpt_geo_dict, coord_geo_dict,
                                                                       max_coord_id, ddr_version, dict_eobt,excluded_airports)
        add_trajectory_scr(engine,data)
        dict_scr=read_trajectory(engine,"scr",flight_id_list=list(data['flight_id']))
        if scr_point is not None:
            scr_point['trajectory_id']=scr_point['flight_id'].apply(lambda x: dict_scr.get(x))
            add_trajectory_points(engine,scr_point[['trajectory_id','geopoint_id','distance','fl','time_over','type',
                                             'rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_srr:
        if verbose:
            print("Points SRR")
        srr_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'srrAllFtPointProfile',engine,
                                                                       airports_geo_dict, wpt_geo_dict, coord_geo_dict,
                                                                       max_coord_id,ddr_version, dict_eobt,excluded_airports)
        add_trajectory_srr(engine,data)
        dict_srr=read_trajectory(engine,"srr",flight_id_list=list(data['flight_id']))
        if srr_point is not None:
            srr_point['trajectory_id']=srr_point['flight_id'].apply(lambda x: dict_srr.get(x))
            add_trajectory_points(engine,srr_point[['trajectory_id','geopoint_id','distance','fl','time_over',
                                             'type','rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_sur:
        if verbose:
            print("Points SUR")
        sur_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'surAllFtPointProfile',engine,
                                                                       airports_geo_dict, wpt_geo_dict, coord_geo_dict,
                                                                       max_coord_id, ddr_version, dict_eobt,excluded_airports)
        add_trajectory_sur(engine,data)
        dict_sur=read_trajectory(engine,"sur",flight_id_list=list(data['flight_id']))
        if sur_point is not None:
            sur_point['trajectory_id']=sur_point['flight_id'].apply(lambda x: dict_sur.get(x))
            add_trajectory_points(engine,sur_point[['trajectory_id','geopoint_id','distance','fl','time_over',
                                                  'type','rel_dist','visible','order']].replace(r'', np.nan, regex=True))    

    if load_dct:
        if verbose:
            print("Points DCT")
        dct_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'dctAllFtPointProfile',engine,
                                                                       airports_geo_dict, wpt_geo_dict, coord_geo_dict, 
                                                                       max_coord_id, ddr_version, dict_eobt,excluded_airports)
        add_trajectory_dct(engine,data)
        dict_dct=read_trajectory(engine,"dct",flight_id_list=list(data['flight_id']))
        if dct_point is not None:
            dct_point['trajectory_id']=dct_point['flight_id'].apply(lambda x: dict_dct.get(x))
            add_trajectory_points(engine,dct_point[['trajectory_id','geopoint_id','distance','fl','time_over',
                                             'type','rel_dist','visible','order']].replace(r'', np.nan, regex=True))

    if load_cpf:
        if verbose:
            print("Points CPF")
        cpf_point, coord_geo_dict, max_coord_id=dataframe_pointProfile(data,'cpfAllFtPointProfile',engine, 
                                                                       airports_geo_dict, wpt_geo_dict, coord_geo_dict, 
                                                                       max_coord_id, ddr_version, dict_eobt,excluded_airports)
        add_trajectory_cpf(engine,data)
        dict_cpf=read_trajectory(engine,"cpf",flight_id_list=list(data['flight_id']))
        if cpf_point is not None:
            cpf_point['trajectory_id']=cpf_point['flight_id'].apply(lambda x: dict_cpf.get(x))
            add_trajectory_points(engine,cpf_point[['trajectory_id','geopoint_id','distance','fl','time_over',
                                                  'type','rel_dist','visible','order']].replace(r'', np.nan, regex=True))
                                       
    return coord_geo_dict, max_coord_id, dict_ftfm, dict_rtfm, dict_ctfm, dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf



def read_trajectory(engine,ttype,flight_id_list=None):
    sql="SELECT t.id, t.flight_id \
        FROM trajectory t WHERE type=\""+str(ttype)+"\""

    if flight_id_list is not None:
      sql = sql+" AND t.flight_id in ("+str(flight_id_list).replace("[","").replace("]","")+")"

    df = pd.read_sql(sql, engine)

    trajectory_dict = pd.Series(df.id.values,index=df.flight_id).to_dict()
    return trajectory_dict



def set_foreing_key_checks(conn,fk):
    conn.execute("SET foreign_key_checks="+str(fk))



def add_flight_airspaces(engine,data, airsp_dict, sect_dict, coord_geo_dict, 
                             dict_ftfm, dict_rtfm, dict_ctfm, dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf,
                             max_coord_id=-1,ddr_version=3,verbose=False,
                             load_ftfm=True, load_rtfm=True, load_ctfm=True, load_scr=True,
                             load_srr=True, load_sur=True, load_dct=True, load_cpf=True):

    if ddr_version==2:
        dict_aobt=pd.Series(data.aobt.values,index=data.flight_id).to_dict()
        dict_cobt=pd.Series(data.cobt.values,index=data.flight_id).to_dict()
        dict_lobt=pd.Series(data.lobt.values,index=data.flight_id).to_dict()
        dict_eobt=pd.Series(data.eobt.values,index=data.flight_id).to_dict()
    else:
        dict_aobt={}
        dict_cobt={}
        dict_lobt={}
        dict_eobt={}


    if load_ftfm:
        if verbose:
            print("Airspace FTFM")
            
        ftfm_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'ftfmAirspProfile', engine, 
                                                airsp_dict, sect_dict, dict_ftfm, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)
        
        if ftfm_airspace is not None:
            add_trajectory_sectors(engine,ftfm_airspace[ ~ftfm_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,ftfm_airspace[ ~ftfm_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    if load_rtfm:
        if verbose:
            print("Airspace RTFM")
        rtfm_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'rtfmAirspProfile', engine, 
                                                airsp_dict, sect_dict, dict_rtfm, coord_geo_dict, max_coord_id, ddr_version, dict_cobt)
        if rtfm_airspace is not None:
            add_trajectory_sectors(engine,rtfm_airspace[ ~rtfm_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,rtfm_airspace[ ~rtfm_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))
        
    if load_ctfm:
        if verbose:
            print("Airspace CTFM")
        ctfm_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'ctfmAirspProfile', engine, 
                                                airsp_dict, sect_dict, dict_ctfm, coord_geo_dict, max_coord_id, ddr_version, dict_aobt)
        if ctfm_airspace is not None:
            add_trajectory_sectors(engine,ctfm_airspace[ ~ctfm_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,ctfm_airspace[ ~ctfm_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))


    if load_scr:        
        if verbose:
            print("Airspace SCR")
        scr_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'scrAirspProfile', engine, 
                                               airsp_dict, sect_dict, dict_scr, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)

        if scr_airspace is not None:
            add_trajectory_sectors(engine,scr_airspace[ ~scr_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,scr_airspace[ ~scr_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    if load_srr:
        if verbose:
            print("Airspace SRR")
        srr_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'srrAirspProfile', engine, 
                                               airsp_dict, sect_dict, dict_srr, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)
        if srr_airspace is not None:
            add_trajectory_sectors(engine,srr_airspace[ ~srr_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,srr_airspace[ ~srr_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    if load_sur:
        if verbose:
            print("Airspace SUR")
        sur_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'surAirspProfile', engine, 
                                               airsp_dict, sect_dict, dict_sur, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)
        if sur_airspace is not None:
            add_trajectory_sectors(engine,sur_airspace[ ~sur_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,sur_airspace[ ~sur_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    if load_dct:
        if verbose:
            print("Airspace DCT")
        dct_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'dctAirspProfile', engine, 
                                               airsp_dict, sect_dict, dict_dct, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)
        if dct_airspace is not None:
            add_trajectory_sectors(engine,dct_airspace[ ~dct_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,dct_airspace[ ~dct_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    if load_cpf:
        if verbose:
            print("Airspace CPF")
        cpf_airspace, coord_geo_dict, max_coord_id=dataframe_airspaceProfile(data,'cpfAirspProfile', engine, 
                                               airsp_dict, sect_dict, dict_cpf, coord_geo_dict, max_coord_id, ddr_version, dict_eobt)
        if cpf_airspace is not None:
            add_trajectory_sectors(engine,cpf_airspace[ ~cpf_airspace['sector_id'].isnull() ].replace(r'', np.nan, regex=True)) 
            add_trajectory_airspaces(engine,cpf_airspace[ ~cpf_airspace['airspace_id'].isnull() ].replace(r'', np.nan, regex=True))

    return coord_geo_dict, max_coord_id


def add_flight_circles_intersections(engine,data, coord_geo_dict, 
                                         dict_ftfm, dict_rtfm, dict_ctfm, dict_scr, dict_srr, dict_sur, dict_dct, dict_cpf,
                                         airports_dict,
                                         max_coord_id, ddr_version=3,verbose=False,
                                         load_ftfm=True, load_rtfm=True, load_ctfm=True, load_scr=True,
                                         load_srr=True, load_sur=True, load_dct=True, load_cpf=True):

    if ddr_version==2:
        dict_aobt=pd.Series(data.aobt.values,index=data.flight_id).to_dict()
        dict_cobt=pd.Series(data.cobt.values,index=data.flight_id).to_dict()
        dict_lobt=pd.Series(data.lobt.values,index=data.flight_id).to_dict()
        dict_eobt=pd.Series(data.eobt.values,index=data.flight_id).to_dict()
    else:
        dict_aobt={}
        dict_cobt={}
        dict_lobt={}
        dict_eobt={}
        

    if load_ftfm:
        ftfm_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'ftfmAllFtCircleIntersections',
                                                                                      engine,dict_ftfm,coord_geo_dict,
                                                                                      airports_dict,
                                                                                      max_coord_id,ddr_version,dict_eobt)
        if ftfm_circle is not None:
            add_trajectory_circles_intersections(engine,ftfm_circle.replace(r'', np.nan, regex=True))
        
    if load_rtfm:
        rtfm_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'rtfmAllFtCircleIntersections',
                                                                                      engine,dict_rtfm,coord_geo_dict,
                                                                                      airports_dict,
                                                                                      max_coord_id,ddr_version,dict_cobt)
        if rtfm_circle is not None:
            add_trajectory_circles_intersections(engine,rtfm_circle.replace(r'', np.nan, regex=True))  
        
    if load_ctfm:
        ctfm_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'ctfmAllFtCircleIntersections',
                                                                                      engine,dict_ctfm,coord_geo_dict,
                                                                                      airports_dict,
                                                                                      max_coord_id,ddr_version,dict_aobt)
        if ctfm_circle is not None:
            add_trajectory_circles_intersections(engine,ctfm_circle.replace(r'', np.nan, regex=True))
        
    if load_scr:
        scr_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'scrAllFtCircleIntersections',
                                                                                     engine,dict_scr,coord_geo_dict,
                                                                                     airports_dict,
                                                                                     max_coord_id,ddr_version,dict_eobt)
        if scr_circle is not None:
            add_trajectory_circles_intersections(engine,scr_circle.replace(r'', np.nan, regex=True))
        
    if load_srr:
        srr_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'srrAllFtCircleIntersections',
                                                                                     engine,dict_srr,coord_geo_dict,
                                                                                     airports_dict,
                                                                                     max_coord_id,ddr_version,dict_eobt)
        if srr_circle is not None:
            add_trajectory_circles_intersections(engine,srr_circle.replace(r'', np.nan, regex=True))
        
    if load_sur:
        sur_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'surAllFtCircleIntersections',
                                                                                     engine,dict_sur,coord_geo_dict,
                                                                                     airports_dict,
                                                                                     max_coord_id,ddr_version,dict_eobt)
        if sur_circle is not None:
            add_trajectory_circles_intersections(engine,sur_circle.replace(r'', np.nan, regex=True))
        
    if load_dct:
        dct_circle, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'dctAllFtCircleIntersections',
                                                                                     engine,dict_dct,coord_geo_dict,
                                                                                     airports_dict,
                                                                                     max_coord_id,ddr_version,dict_eobt)        
        if dct_circle is not None:
            add_trajectory_circles_intersections(engine,dct_circle.replace(r'', np.nan, regex=True))
        
    if load_cpf:
        cpf_airspace, coord_geo_dict, max_coord_id=dataframe_circleIntersectionProfile(data,'cpfAllFtCircleIntersections',
                                                                                       engine,dict_cpf,coord_geo_dict,
                                                                                       airports_dict,
                                                                                       max_coord_id,ddr_version,dict_eobt)        
        if cpf_airspace is not None:
            add_trajectory_circles_intersections(engine,cpf_airspace.replace(r'', np.nan, regex=True))

    return coord_geo_dict, max_coord_id


def add_flight_fl_requests(engine, data, dict_ftfm, dict_rtfm, dict_ctfm,verbose=False,
                           load_ftfm=True, load_rtfm=True, load_ctfm=True):

    if load_ftfm:
        ftfm_speed_request=dataframe_requestSpeedsProfile(data,'ftfmReqFLSpeedList',dict_ftfm)
        if ftfm_speed_request is not None:
            add_trajectory_speed_request(engine,ftfm_speed_request.replace(r'', np.nan, regex=True))
            
    if load_rtfm:
        rtfm_speed_request=dataframe_requestSpeedsProfile(data,'rtfmReqFLSpeedList',dict_rtfm)
        if rtfm_speed_request is not None:
            add_trajectory_speed_request(engine,rtfm_speed_request.replace(r'', np.nan, regex=True))

    if load_ctfm:
        ctfm_speed_request=dataframe_requestSpeedsProfile(data,'ctfmReqFLSpeedList',dict_ctfm)
        if ctfm_speed_request is not None:
            add_trajectory_speed_request(engine,ctfm_speed_request.replace(r'', np.nan, regex=True))



def add_trajectory_speed_request(engine,data):
        trajectory_speed_request_field=['trajectory_id','fl','speed','speed_units','distance']
        data[trajectory_speed_request_field].to_sql("trajectory_fl_request", engine, index=False, if_exists="append")
     

def add_eets(engine,data,sect_dict,wpt_geo_dict,airports_geo_dict,coord_geo_dict,dict_ftfm,max_coord_id,verbose=False):
    ftfm_eet_fir=dataframe_eet_fir(data,dict_ftfm,sect_dict)
    if ftfm_eet_fir is not None:
        add_eet_fir(engine,ftfm_eet_fir.replace(r'', np.nan, regex=True))
        
    ftfm_eet_point, coord_geo_dict, max_coord_id=dataframe_eet_points(data,dict_ftfm,
                                                                      wpt_geo_dict,airports_geo_dict,
                                                                      coord_geo_dict,max_coord_id,engine)
    if ftfm_eet_point is not None:
        add_eet_points(engine,ftfm_eet_point.replace(r'', np.nan, regex=True))
    
    return coord_geo_dict, max_coord_id


def add_eet_fir(engine,data):
    eet_fir_field=['trajectory_id','fir_sid','fir_id','eet','order']
    print(data[data['fir_id'].isnull()])
    data[eet_fir_field].to_sql("trajectory_eet_fir", engine, index=False, if_exists="append")

def add_eet_points(engine,data):
    eet_fir_field=['trajectory_id','geopoint_id','eet','point_sid','order']
    data[eet_fir_field].to_sql("trajectory_eet_point", engine, index=False, if_exists="append")


def delete_flights_info_db(engine,ifps_ids=None, tact_ids=None):
    if ifps_ids is not None:
        if len(ifps_ids)==0:
            ifps_ids=None

    if tact_ids is not None:
        if len(tact_ids)==0:
            tact_ids=None
            

    if (ifps_ids is not None and len(ifps_ids)>0) or \
       (tact_ids is not None and len(tact_ids)>0):
        delete_trajectories(engine,ifps_ids,tact_ids)
        delete_flights(engine,ifps_ids,tact_ids)

def delete_trajectories(engine,ifps_ids=None, tact_ids=None):

        delete_trajectories_elements(engine,"trajectory_has_sector",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_has_airspace",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_eet_point",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_eet_fir",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_fl_request",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_has_geopoint",ifps_ids,tact_ids)
        delete_trajectories_elements(engine,"trajectory_has_circle_intersections",ifps_ids,tact_ids)
        

        sql = "delete t from trajectory t "+\
              "join flight f on f.id = t.flight_id "

        if (ifps_ids is not None) or (tact_ids is not None):
            sql = sql + "where "

            if ifps_ids is not None:
                sql = sql + "f.ifps_id IN "+str(ifps_ids).replace('[','(').replace(']',')')

            if tact_ids is not None:
                if ifps_ids is not None:
                    sql=sql + " OR "
                sql = sql + "f.tact_id IN "+str(tact_ids).replace('[','(').replace(']',')')
            
        #conn = engine.connect()
        engine.execute(sql)
        #conn.close()


def delete_trajectories_elements(engine,table,ifps_ids=None,tact_ids=None):

    sql="delete te from "+table+" te "+\
        "join trajectory t on t.id = te.trajectory_id "+\
        "join flight f on f.id = t.flight_id "

    if (ifps_ids is not None) or (tact_ids is not None):
        sql = sql + "where "

        if ifps_ids is not None:
            sql = sql + "f.ifps_id IN "+str(ifps_ids).replace('[','(').replace(']',')')

        if tact_ids is not None:
            if ifps_ids is not None:
                sql=sql + " OR "
            sql = sql + "f.tact_id IN "+str(tact_ids).replace('[','(').replace(']',')')
        
    #conn = engine.connect()
    engine.execute(sql)
    #conn.close()


def delete_flights(engine,ifps_ids=None,tact_ids=None):
        delete_flight_element(engine,"flight_intention_info")
        delete_flight_element(engine,"flight_cdm_info")
        delete_flight_element(engine,"flight_atfm_info")

        sql = "delete f from flight f "

        if (ifps_ids is not None) or (tact_ids is not None):
            sql = sql + "where "

            if ifps_ids is not None:
                sql = sql + "f.ifps_id IN "+str(ifps_ids).replace('[','(').replace(']',')')

            if tact_ids is not None:
                if ifps_ids is not None:
                    sql=sql + " OR "
                sql = sql + "f.tact_id IN "+str(tact_ids).replace('[','(').replace(']',')')
            
        #conn = engine.connect()
        engine.execute(sql)
        #conn.close()


def delete_flight_element(engine,table,ifps_ids=None, tact_ids=None):

    sql="delete fe from "+table+" fe "+\
        "join flight f on f.id = fe.flight_id "

    if (ifps_ids is not None) or (tact_ids is not None):
        sql = sql + "where "

        if ifps_ids is not None:
            sql = sql + "f.ifps_id IN "+str(ifps_ids).replace('[','(').replace(']',')')

        if tact_ids is not None:
            if ifps_ids is not None:
                sql=sql + " OR "
            sql = sql + "f.tact_id IN "+str(tact_ids).replace('[','(').replace(']',')')
        
    #conn = engine.connect()
    engine.execute(sql)
    #conn.close()




def add_trajectory_ftfm(engine,data):
    ftfm_database_fields_orig=['flight_id', 'ftfmAiracCycleRelease', 'ftfmEvnBaselineNumber', 'ftfmDepRunway', 'ftfmArrRunway',
          'ftfmConsumedFuel', 'ftfmRouteCharges', 'ftfmAllFtPointProfile']
    
    d=data[ftfm_database_fields_orig].dropna(thresh=2).copy()
    
    
    ftfm_database_fields_orig=['flight_id', 'ftfmAiracCycleRelease', 'ftfmEvnBaselineNumber', 'ftfmDepRunway', 'ftfmArrRunway',
          'ftfmConsumedFuel', 'ftfmRouteCharges']
    
    d=d[ftfm_database_fields_orig].copy()
    
    
    ftfm_database_fields=['flight_id', 'airac_cycle_release', 'env_baseline_number', 'departure_runway', 'arrival_runway',
                          'fuel_consumed', 'crco', 'type','obt']
    
    d['type']='ftfm'
    d['obt']=np.nan
    
    d.columns=ftfm_database_fields
    
    
    d[ftfm_database_fields].to_sql("trajectory",engine, index=False, if_exists="append")



def add_trajectory_rtfm(engine,data):
    rtfm_database_fields_orig=['flight_id', 'rtfmAiracCycleRelease', 'rtfmEvnBaselineNumber', 'rtfmDepRunway', 'rtfmArrRunway',
          'rtfmConsumedFuel', 'rtfmRouteCharges', 'rtfmAllFtPointProfile']
    
    d=data[rtfm_database_fields_orig].dropna(thresh=2).copy()
    
    rtfm_database_fields_orig=['flight_id', 'rtfmAiracCycleRelease', 'rtfmEvnBaselineNumber', 'rtfmDepRunway', 'rtfmArrRunway',
          'rtfmConsumedFuel', 'rtfmRouteCharges']
    
    d=d[rtfm_database_fields_orig].copy()
    
    
    rtfm_database_fields=['flight_id', 'airac_cycle_release', 'env_baseline_number', 'departure_runway', 'arrival_runway',
                          'fuel_consumed', 'crco', 'type', 'obt']
    
    d['type']='rtfm'
    d['obt']=np.nan
    
    d.columns=rtfm_database_fields

    
    d[rtfm_database_fields].to_sql("trajectory",engine, index=False, if_exists="append")


def add_trajectory_ctfm(engine,data):
    ctfm_database_fields_orig=['flight_id', 'ctfmAiracCycleRelease', 'ctfmEvnBaselineNumber', 'ctfmDepRunway', 'ctfmArrRunway',
          'ctfmConsumedFuel', 'ctfmRouteCharges', 'ctfmAllFtPointProfile']
    
    d=data[ctfm_database_fields_orig].dropna(thresh=2).copy()
    
    ctfm_database_fields_orig=['flight_id', 'ctfmAiracCycleRelease', 'ctfmEvnBaselineNumber', 'ctfmDepRunway', 'ctfmArrRunway',
          'ctfmConsumedFuel', 'ctfmRouteCharges']
    
    d=d[ctfm_database_fields_orig].copy()
    
    
    ctfm_database_fields=['flight_id', 'airac_cycle_release', 'env_baseline_number', 'departure_runway', 'arrival_runway',
                          'fuel_consumed', 'crco', 'type','obt']
    
    d['type']='ctfm'
    d['obt']=np.nan
    
    d.columns=ctfm_database_fields

    
    d[ctfm_database_fields].to_sql("trajectory",engine, index=False, if_exists="append")


def add_trajectory_srr(engine,data):
    srr_database_fields_orig=['flight_id', 'srrConsumedFuel', 'srrRouteCharges', 'srrObt', 'srrAllFtPointProfile']
    d=data[srr_database_fields_orig].dropna(thresh=2).copy()
    srr_database_fields_orig=['flight_id', 'srrConsumedFuel', 'srrRouteCharges', 'srrObt']
    d=d[srr_database_fields_orig].copy()
    trajectory_db_fields=['flight_id', 'fuel_consumed', 'crco', 'obt', 'type']
    d['type']='srr'
    d.columns=trajectory_db_fields

    
    d[trajectory_db_fields].to_sql("trajectory",engine, index=False, if_exists="append")
    
    

def add_trajectory_scr(engine,data):
    scr_database_fields_orig=['flight_id', 'scrConsumedFuel', 'scrRouteCharges', 'scrObt', 'scrAllFtPointProfile']
    d=data[scr_database_fields_orig].dropna(thresh=2).copy()
    scr_database_fields_orig=['flight_id', 'scrConsumedFuel', 'scrRouteCharges', 'scrObt']
    d=d[scr_database_fields_orig].copy()
    trajectory_db_fields=['flight_id', 'fuel_consumed', 'crco', 'obt', 'type']
    d['type']='scr'
    d.columns=trajectory_db_fields

    
    d[trajectory_db_fields].to_sql("trajectory",engine, index=False, if_exists="append")



def add_trajectory_cpf(engine,data):
        cpf_database_fields_orig=['flight_id', 'cpfConsumedFuel', 'cpfRouteCharges', 'cpfObt','cpfAllFtPointProfile']
        d=data[cpf_database_fields_orig].dropna(thresh=2).copy()
        cpf_database_fields_orig=['flight_id', 'cpfConsumedFuel', 'cpfRouteCharges', 'cpfObt']
        d=d[cpf_database_fields_orig].copy()
        trajectory_db_fields=['flight_id', 'fuel_consumed', 'crco', 'obt', 'type']
        d['type']='cpf'
        d.columns=trajectory_db_fields
        
        d[trajectory_db_fields].to_sql("trajectory",engine, index=False, if_exists="append")


def add_trajectory_dct(engine,data): 
        dct_database_fields_orig=['flight_id', 'dctConsumedFuel', 'dctRouteCharges', 'dctObt', 'dctAllFtPointProfile']
        d=data[dct_database_fields_orig].dropna(thresh=2).copy()
        dct_database_fields_orig=['flight_id', 'dctConsumedFuel', 'dctRouteCharges', 'dctObt']
        d=d[dct_database_fields_orig].copy()
        trajectory_db_fields=['flight_id', 'fuel_consumed', 'crco', 'obt', 'type']
        d['type']='dct'
        d.columns=trajectory_db_fields

        
        d[trajectory_db_fields].to_sql("trajectory",engine, index=False, if_exists="append")
        

def add_trajectory_sur(engine,data):
    sur_database_fields_orig=['flight_id', 'surConsumedFuel', 'surRouteCharges', 'surObt', 'surAllFtPointProfile']
    d=data[sur_database_fields_orig].dropna(thresh=2).copy()
    sur_database_fields_orig=['flight_id', 'surConsumedFuel', 'surRouteCharges', 'surObt']
    d=d[sur_database_fields_orig].copy()
    trajectory_db_fields=['flight_id', 'fuel_consumed', 'crco', 'obt', 'type']
    d['type']='sur'
    d.columns=trajectory_db_fields

    
    d[trajectory_db_fields].to_sql("trajectory",engine, index=False, if_exists="append")


def add_trajectory_points(engine,data):
    trajectory_has_geopoints_fields=['trajectory_id','geopoint_id','distance','fl','time_over','type','rel_dist','visible','order']; 
    #load_data_infile(data[trajectory_has_geopoints_fields], 'trajectory_has_geopoint')
    increment=500000 
    nelems=len(data) 
    le=0 
    ue=min(increment,nelems) 
     
    while le<nelems: 
        #insert 
        #load_data_infile(data.iloc[le:ue,data.columns.get_indexer(trajectory_has_geopoints_fields)],'trajectory_has_geopoint')
        data.iloc[le:ue,data.columns.get_indexer(trajectory_has_geopoints_fields)].to_sql("trajectory_has_geopoint", engine, index=False, if_exists="append") 
        le=ue 
        #ue=min(ue+increment,nelems) 
        ue=ue+increment 

def extract_lat(x):
        index_lat=max(x.find('N'), x.find('S'))

        deg=Decimal(x[0:2])
        if index_lat>=4:
            minutes=Decimal(x[2:4])
        else:
            minutes=Decimal(0)
        if index_lat>=6:
            seconds=Decimal(x[4:6])
        else:
            seconds=Decimal(0)

        lat=deg+minutes/60+seconds/60/60
        if x.find('S')>0:
            lat=-(deg+minutes/60+seconds/60/60)
            
        return lat
        
def extract_lon(x):
    index_lat=max(x.find('N'), x.find('S'))
    deg=Decimal(x[index_lat+1:index_lat+4])
    if index_lat+5<len(x):
        minutes=Decimal(x[index_lat+4:index_lat+6])
    else:
        minutes=Decimal(0)

    if index_lat+7<len(x):
        seconds=Decimal(x[index_lat+6:index_lat+8])
    else:
        seconds=Decimal(0)

    lon=deg+minutes/60+seconds/60/60
    if x.find('W')>0:
        lon=-(deg+minutes/60+seconds/60/60)

    return lon



def add_missing_coordinate_geopoints(engine,missing_geo_points):
        max_geo_id=read_maxid_geopoints(engine)
        max_coord_id=read_maxid_coordpoints(engine)
        
        add_coordpoints(engine,missing_geo_points)
        coord_dict=read_coordpoints(engine,max_coord_id)

        missing_geo_points['lat']=missing_geo_points['sid'].apply(extract_lat)
        missing_geo_points['lon']=missing_geo_points['sid'].apply(extract_lon)

        missing_geo_points['latlon']=missing_geo_points["lat"].map(str) +"_"+ missing_geo_points["lon"].map(str)
        missing_geo_points['type']="GEO"
        add_geopoints(engine,missing_geo_points)
        geopoints=read_geopoints_ids(engine,from_p_id=max_geo_id,rounded=1)

        missing_geo_points['coordpoint_id']=missing_geo_points['sid'].apply(lambda x: coord_dict.get(x))
        missing_geo_points['lat_rounded']=missing_geo_points['lat'].apply(lambda x: str(truncate(x,12)))#str(int(x * 10**12)/ 10**12))
        missing_geo_points['lon_rounded']=missing_geo_points['lon'].apply(lambda x: str(truncate(x,12)))#str(int(x * 10**12)/ 10**12))

        missing_geo_points['lat_lon']=missing_geo_points["lat_rounded"] +"_"+ missing_geo_points["lon_rounded"]
        missing_geo_points['geopoint_id']=missing_geo_points['lat_lon'].apply(lambda x: geopoints.get(x)['id'])

        add_coordpoint_point_connection(engine,missing_geo_points[['coordpoint_id','geopoint_id']])


def read_maxid_geopoints(engine):
    sql="SELECT max(g.id) as max_id FROM geopoint g"
    df = pd.read_sql(sql, engine)
    max_id=df.loc[0,'max_id']
    if max_id is None:
        max_id=0
    return max_id


def add_coordpoints(engine,data):
    #load_data_infile(pd[['sid']], 'coordpoint')
    increment=500000 
    nelems=len(data) 
    le=0 
    ue=min(increment,nelems) 
     
    while le<nelems: 
        #insert 
        data.iloc[le:ue,data.columns.get_indexer(['sid'])].to_sql("coordpoint", engine, index=False, if_exists="append") 
        le=ue 
        ue=ue+increment


def read_coordpoints(engine, from_c_id=-1, sid=None):
    sql="SELECT c.id, c.sid \
         FROM coordpoint c WHERE c.id> "+str(from_c_id)

    if sid is not None:
      sql = sql + " AND c.sid in ("+str(sid).replace('[','').replace(']','')+")"

    df = pd.read_sql(sql, engine)

    coord_dict = pd.Series(df.id.values,index=df.sid).to_dict()

    return coord_dict

def read_coordpoints_with_geopoints(engine, from_c_id=-1, sid=None):
    sql= "SELECT c.id as coord_id, c.sid as sid, g.id as geo_id \
        FROM coordpoint c \
        JOIN coordpoint_has_geopoint chg on chg.coordpoint_id=c.id \
        JOIN geopoint g on g.id=chg.geopoint_id \
        WHERE c.id> "+str(from_c_id);

    if sid is not None:
      sql = sql + " AND c.sid in ("+str(sid).replace('[','').replace(']','')+")"

    df = pd.read_sql(sql, engine)

    coord_dict=df.set_index('sid')[['coord_id','geo_id']].to_dict('index')

    #for row in cur:
    #    coord_dict[row[1]]={'coord_id':row[0], 'geo_id':row[2]};
    #conn.close()
    
    return coord_dict

def add_trajectory_sectors(engine,data):
    trajectory_has_sector_fields=['trajectory_id','sector_id','geopoint_entry_id','geopoint_exit_id','fl_entry','fl_exit',
                                  'distance_entry','distance_exit','time_entry','time_exit','order']
    data[trajectory_has_sector_fields].to_sql("trajectory_has_sector", engine, index=False, if_exists="append")
    
def add_trajectory_airspaces(engine,data):
    trajectory_has_sector_fields=['trajectory_id','airspace_id','geopoint_entry_id','geopoint_exit_id','fl_entry','fl_exit',
                                  'distance_entry','distance_exit','time_entry','time_exit','order']
    data[trajectory_has_sector_fields].to_sql("trajectory_has_airspace", engine, index=False, if_exists="append")

def add_trajectory_circles_intersections(engine,data):
    circles_intersection_fields=['trajectory_id','type','from_last','to_from_airport_id','geopoint_entry_id','geopoint_exit_id',
                                'fl_entry','fl_exit','distance_radius_nm','time_entry','time_exit','distance_entry','distance_exit']
    data[circles_intersection_fields].to_sql("trajectory_has_circle_intersections", engine, index=False, if_exists="append")


def add_coordpoint_point_connection(engine,data):
    #load_data_infile(pd[['coordpoint_id','geopoint_id']], 'coordpoint_has_geopoint')
    increment=500000 
    nelems=len(data) 
    le=0 
    ue=min(increment,nelems) 
     
    while le<nelems: 
        #insert 
        data.iloc[le:ue,data.columns.get_indexer(['coordpoint_id','geopoint_id'])].to_sql("coordpoint_has_geopoint", engine, index=False, if_exists="append") 
        le=ue 
        ue=ue+increment 


def zip_with_scalar_divide_point(l, o):
    return ((o, i.split(":")) for i in l)

    
def dataframe_pointProfile(data, pointName, engine, airports_geo_dict, wpt_geo_dict, 
                           coord_geo_dict, max_coord_id=-1, ddr_version=3,dict_date={}, excluded_airports=[""]):

    read_points_on_demand = (len(coord_geo_dict)==0)#read points on demand
    
    if len(data.loc[(~pd.isnull(data[pointName])),:])>0:
        d=(data.loc[(~pd.isnull(data[pointName])),:].apply(lambda x: zip_with_scalar_divide_point(x[pointName].split(" "),x['flight_id']) if 
                      not pd.isnull(x[pointName]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','allFtPoint']

        g=f[['flight_id']].copy()

        g['timeover_orig']=f['allFtPoint'].apply(lambda x: x[0])
        g['point']=f['allFtPoint'].apply(lambda x: x[1])
        g['route']=f['allFtPoint'].apply(lambda x: x[2])
        g['fl']=f['allFtPoint'].apply(lambda x: x[3])
        g['distance']=f['allFtPoint'].apply(lambda x: x[4])
        g['type']=f['allFtPoint'].apply(lambda x: x[5])
        g['geopoint']=f['allFtPoint'].apply(lambda x: x[6])
        g['rel_dist']=f['allFtPoint'].apply(lambda x: x[7])
        g['visible']=f['allFtPoint'].apply(lambda x: 0 if x[8]=="N" else 1)
        
        g['geopoint_id']=g.apply(lambda x: airports_geo_dict.get(x['point'])['geo_id']
                                 if not pd.isnull(airports_geo_dict.get(x['point'],np.nan)) 
                                    and x['point'] not in excluded_airports
                                 else -1 if x['point'] in excluded_airports and x['geopoint']==""
                                 else wpt_geo_dict.get(x['point'])['geo_id'] 
                                 if not pd.isnull(wpt_geo_dict.get(x['point'],np.nan))
                                 else coord_geo_dict.get(x['geopoint'])['geo_id'] 
                                 if not pd.isnull(coord_geo_dict.get(x['geopoint'],np.nan))
                                 else np.nan, axis=1)


        missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_id']),['geopoint']].geopoint.unique())
        missing_geo_points.columns=['sid']

        if read_points_on_demand:
          dict_coord_points_extra = read_coordpoints_with_geopoints(engine,sid=list(missing_geo_points['sid']))
          coord_geo_dict.update(dict_coord_points_extra)

          g['geopoint_id']=g.apply(lambda x: airports_geo_dict.get(x['point'])['geo_id']
                                 if not pd.isnull(airports_geo_dict.get(x['point'],np.nan)) 
                                    and x['point'] not in excluded_airports
                                 else -1 if x['point'] in excluded_airports and x['geopoint']==""
                                 else wpt_geo_dict.get(x['point'])['geo_id'] 
                                 if not pd.isnull(wpt_geo_dict.get(x['point'],np.nan))
                                 else coord_geo_dict.get(x['geopoint'])['geo_id'] 
                                 if not pd.isnull(coord_geo_dict.get(x['geopoint'],np.nan))
                                 else np.nan, axis=1)

          missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_id']),['geopoint']].geopoint.unique())
          missing_geo_points.columns=['sid']

        indexes_missing=pd.isnull(g['geopoint_id'])

        if not missing_geo_points.empty:
            #need to add points
            max_coord_id=read_maxid_coordpoints(engine)
            add_missing_coordinate_geopoints(engine,missing_geo_points)
            
            coord_geo_dict_extra=read_coordpoints_with_geopoints(engine,max_coord_id)
            coord_geo_dict.update(coord_geo_dict_extra)
            max_coord_id=read_maxid_coordpoints(engine)

            g.loc[indexes_missing,['geopoint_id']]=g.loc[indexes_missing]['geopoint'].apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else x)
            
            

        if ddr_version >= 3:
            g['time_over']=g['timeover_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
                                                    x[8:10]+":"+x[10:12]+":"+x[12:14])
            
        else:            
            g['date_ini']=f['flight_id'].apply(lambda x: datetime.strptime(dict_date.get(x)[0:10], '%Y-%m-%d'))
            g['hour_ini']=g['flight_id'].apply(lambda x: int(dict_date.get(x)[11:13]))
            g['time_over_orig_formated']=g['timeover_orig'].apply(lambda x: np.nan if x=='' 
                                                                     else x[0:2]+":"+x[2:4]+":"+x[4:6])
            #in some cases the timeover is "" like in flight AA34925529 / flihgt_id=39429 which starts at ZZZZ
            
            
            g['time_over']=g.apply(lambda x: 
                                  np.nan if x['timeover_orig']==''
                                  else
                                  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_over_orig_formated']
                                  if int(x['timeover_orig'][0:2])>=x['hour_ini'] else
                                  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_over_orig_formated']
                                  , axis=1)
            
            g['change_day']=g.apply(lambda x: 
                                  np.nan if x['timeover_orig']==''
                                  else
                                  0
                                  if int(x['timeover_orig'][0:2])>=x['hour_ini'] else
                                  1
                                  , axis=1)
        

        g = g[g['geopoint_id']!=-1].reset_index(drop=True).copy()
        g.insert(0, 'order', (g.groupby(['flight_id']).cumcount()+1))
               
    else:
        g=None

    if read_points_on_demand:
      coord_geo_dict = {}

    return g, coord_geo_dict, max_coord_id

    
def dataframe_airspaceProfile(data, airspaceName, engine, airsp_dict, sect_dict, dict_trajectory, coord_geo_dict, max_coord_id, ddr_version=3,dict_date={}):
    #For each airspace field in each flight divide it into a dataframe identified by filght_id and allAirsp information
    
    read_points_on_demand = (len(coord_geo_dict)==0)#read points on demand

    if len(data.loc[(~pd.isnull(data[airspaceName])),:])>0:
            
        d=(data.apply(lambda x: zip_with_scalar_divide_point(x[airspaceName].split(" "),x['flight_id']) if 
                          not pd.isnull(x[airspaceName]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','allAirsp']

        #divide the information of the ALL_FT+ into fields
        g=f[['flight_id']].copy()
        g['trajectory_id']=g['flight_id'].apply(lambda x: dict_trajectory.get(x))    
        g['time_entry_orig']=f['allAirsp'].apply(lambda x: x[0])
        g['airspace']=f['allAirsp'].apply(lambda x: x[1])
        g['time_exit_orig']=f['allAirsp'].apply(lambda x: x[2])
        g['airspace_type']=f['allAirsp'].apply(lambda x: x[3])
        g['latlon_entry']=f['allAirsp'].apply(lambda x: x[4])
        g['latlon_exit']=f['allAirsp'].apply(lambda x: x[5])
        g['fl_entry']=f['allAirsp'].apply(lambda x: x[6] if x[6] != '' else np.nan)
        g['fl_exit']=f['allAirsp'].apply(lambda x: x[7] if x[7] != '' else np.nan)
        g['distance_entry']=f['allAirsp'].apply(lambda x: x[8] if x[8] != '' else np.nan)
        g['distance_exit']=f['allAirsp'].apply(lambda x: x[9] if x[9] != '' else np.nan)


        g['airspace_id']=g.apply(lambda x: airsp_dict.get(x['airspace']) if (x['airspace_type']=="AREA" or
                                                                            x['airspace_type']=="NAS" or
                                                                            x['airspace_type']=="AUA" or
                                                                            x['airspace_type']=="CS" or
                                                                            x['airspace_type']=="CRSA" or
                                                                            x['airspace_type']=="CLUS")
                                                                         else np.NaN, axis=1)

        g['sector_id']=g.apply(lambda x: sect_dict.get(x['airspace']) if (x['airspace_type']=="NS" or 
                                                                            x['airspace_type']=="FIR" or
                                                                            x['airspace_type']=="AOI" or
                                                                            x['airspace_type']=="AOP" or
                                                                            x['airspace_type']=="ES" or
                                                                            x['airspace_type']=="ERSA")
                                                                         else np.NaN, axis=1)

        g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)
        
        

        g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)

        
        missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
        missing_geo_points.columns=['sid']
        
        indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])
        
        missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
        missing_geo_points_exit.columns=['sid']

        indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])
        
        missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')
        
        missing_geo_points.drop_duplicates(inplace=True)

        missing_geo_points=missing_geo_points.loc[missing_geo_points['sid']!='']


        if read_points_on_demand:
          dict_coord_points_extra = read_coordpoints_with_geopoints(engine,sid=list(missing_geo_points['sid']))
          coord_geo_dict.update(dict_coord_points_extra)

          g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)
        

          g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                         else np.nan)

          
          missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
          missing_geo_points.columns=['sid']
          
          indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])
          
          missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
          missing_geo_points_exit.columns=['sid']

          indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])
          
          missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')
          
          missing_geo_points.drop_duplicates(inplace=True)

          missing_geo_points=missing_geo_points.loc[missing_geo_points['sid']!='']

        
        

        if not missing_geo_points.empty:
            #There are geopoints missing
            max_coord_id=read_maxid_coordpoints(engine)
            add_missing_coordinate_geopoints(engine,missing_geo_points)

            coord_geo_dict_extra=read_coordpoints_with_geopoints(engine,max_coord_id)
            coord_geo_dict.update(coord_geo_dict_extra)
            max_coord_id=read_maxid_coordpoints(engine)

            #g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
            #                                       else np.nan)  
            #g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
            #                                       else np.nan)   

            g.loc[indexes_missing_entry,['geopoint_entry_id']]=g.loc[indexes_missing_exit]['latlon_entry']\
                                    .apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else np.nan)

            g.loc[indexes_missing_exit,['geopoint_exit_id']]=g.loc[indexes_missing_exit]['latlon_exit']\
                                    .apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else np.nan)

        

        if ddr_version >= 3:
            g['time_entry']=g['time_entry_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
                                                       x[8:10]+":"+x[10:12]+":"+x[12:14])
            g['time_exit']=g['time_exit_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
                                                     x[8:10]+":"+x[10:12]+":"+x[12:14])
                                                    
        else:
            print(f['flight_id'])
            g['date_ini']=f['flight_id'].apply(lambda x: datetime.strptime(dict_date.get(x)[0:10], '%Y-%m-%d'))
            g['hour_ini']=g['flight_id'].apply(lambda x: int(dict_date.get(x)[11:13]))
            g['time_entry_orig_formated']=g['time_entry_orig'].apply(lambda x: np.nan if x=='' 
                                                                     else x[0:2]+":"+x[2:4]+":"+x[4:6])
            g['time_exit_orig_formated']=g['time_exit_orig'].apply(lambda x: np.nan if x=='' 
                                                                     else x[0:2]+":"+x[2:4]+":"+x[4:6])
            #in some cases the timeover is "" like in flight AA34925529 / flihgt_id=39429 which starts at ZZZZ        
            
            g['time_entry']=g.apply(lambda x: 
                                  np.nan if x['time_entry_orig']==''
                                  else
                                  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
                                  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
                                  , axis=1)
            
            g['change_day_entry']=g.apply(lambda x: 
                                  np.nan if x['time_entry_orig']==''
                                  else
                                  0
                                  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  1
                                  , axis=1)

            
            
            g['time_exit']=g.apply(lambda x: 
                                  np.nan if x['time_exit_orig']==''
                                  else
                                  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
                                  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] and int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
                                  , axis=1)
            

            
            g['change_day_exit']=g.apply(lambda x: 
                                  np.nan if x['time_exit_orig']==''
                                  else
                                  0
                                  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] else
                                  1
                                  , axis=1)
            
                       
        
        g=g[['flight_id','trajectory_id', 'sector_id', 'airspace_id', 'geopoint_entry_id', 'geopoint_exit_id', 'fl_entry', 'fl_exit','distance_entry','distance_exit','time_entry','time_exit']]

        g.insert(0, 'order', (g.groupby(['flight_id']).cumcount()+1))
        
        
        
    else:
        g=None


    if read_points_on_demand:
      coord_geo_dict = {}
            
    return g, coord_geo_dict, max_coord_id


def dataframe_circleIntersectionProfile(data,circleName,engine,dict_trajectory,coord_geo_dict,airports_dict,max_coord_id, ddr_version=3,dict_date={}):
    ####

    read_points_on_demand = (len(coord_geo_dict)==0)#read points on demand
    
    if len(data.loc[(~pd.isnull(data[circleName])),:])>0:
        d=(data.apply(lambda x: zip_with_scalar_divide_point(x[circleName].split(" "),x['flight_id']) if 
                                  not pd.isnull(x[circleName]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','allCircle']

        #divide the information of the ALL_FT+ into fields

        g=f[['flight_id']].copy()
        g['trajectory_id']=g['flight_id'].apply(lambda x: dict_trajectory.get(x))    
        g['time_entry_orig']=f['allCircle'].apply(lambda x: x[0])
        g['circle_info']=f['allCircle'].apply(lambda x: x[1]) #F100EGBE
        g['airport_icao']=f['allCircle'].apply(lambda x: x[1][-4:])
        g['from_last']=f['allCircle'].apply(lambda x: x[1][0])
        g['distance_radius_nm']=f['allCircle'].apply(lambda x: x[1][1:len(x[1])-4])
        g['time_exit_orig']=f['allCircle'].apply(lambda x: x[2])
        g['type']=f['allCircle'].apply(lambda x: x[3])
        g['latlon_entry']=f['allCircle'].apply(lambda x: x[4])
        g['latlon_exit']=f['allCircle'].apply(lambda x: x[5])
        g['fl_entry']=f['allCircle'].apply(lambda x: x[6])
        g['fl_exit']=f['allCircle'].apply(lambda x: x[7])
        g['distance_entry']=f['allCircle'].apply(lambda x: x[8])
        g['distance_exit']=f['allCircle'].apply(lambda x: x[9])
        g['to_from_airport_id']=g['airport_icao'].apply(lambda x: airports_dict.get(x, np.NaN))
        
        g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)   
        g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)       
               

        missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
        missing_geo_points.columns=['sid']
        indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])

        missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
        missing_geo_points_exit.columns=['sid']
        indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])

        missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')
        missing_geo_points.drop_duplicates(inplace=True)


        if read_points_on_demand:
          dict_coord_points_extra = read_coordpoints_with_geopoints(engine,sid=list(missing_geo_points['sid']))
          coord_geo_dict.update(dict_coord_points_extra)
          g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                       else np.nan)   
          g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                                         else np.nan)       
                 

          missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_entry_id']),['latlon_entry']].latlon_entry.unique())
          missing_geo_points.columns=['sid']
          indexes_missing_entry=pd.isnull(g['geopoint_entry_id'])

          missing_geo_points_exit=pd.DataFrame(g.loc[pd.isnull(g['geopoint_exit_id']),['latlon_exit']].latlon_exit.unique())
          missing_geo_points_exit.columns=['sid']
          indexes_missing_exit=pd.isnull(g['geopoint_exit_id'])

          missing_geo_points = pd.merge(missing_geo_points, missing_geo_points_exit, on='sid', how='outer')
          missing_geo_points.drop_duplicates(inplace=True)




        if not missing_geo_points.empty:
            #There are geopoints missing
            max_coord_id=read_maxid_coordpoints(engine)
            add_missing_coordinate_geopoints(engine,missing_geo_points)
            coord_geo_dict_extra=read_coordpoints_with_geopoints(engine,max_coord_id)
            coord_geo_dict.update(coord_geo_dict_extra)
            max_coord_id=read_maxid_coordpoints(engine)
            
            #g['geopoint_entry_id']=g['latlon_entry'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
            #                                       else np.nan)    
            #g['geopoint_exit_id']=g['latlon_exit'].apply(lambda x: coord_geo_dict.get(x)['geo_id'] if not pd.isnull(coord_geo_dict.get(x,np.nan))
            #                                       else np.nan) 

            g.loc[indexes_missing_entry,['geopoint_entry_id']]=g.loc[indexes_missing_exit]['latlon_entry']\
                                     .apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else np.nan)

            g.loc[indexes_missing_exit,['geopoint_exit_id']]=g.loc[indexes_missing_exit]['latlon_exit']\
                                     .apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else np.nan)  
        
        
        if ddr_version >= 3:
            g['time_entry']=g['time_entry_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
                                                       x[8:10]+":"+x[10:12]+":"+x[12:14])
            g['time_exit']=g['time_exit_orig'].apply(lambda x: x[0:4]+"-"+x[4:6]+"-"+x[6:8]+" "+
                                                     x[8:10]+":"+x[10:12]+":"+x[12:14])
                                                    
        else:
            g['date_ini']=f['flight_id'].apply(lambda x: datetime.strptime(dict_date.get(x)[0:10], '%Y-%m-%d'))
            g['hour_ini']=g['flight_id'].apply(lambda x: int(dict_date.get(x)[11:13]))
            g['time_entry_orig_formated']=g['time_entry_orig'].apply(lambda x: np.nan if x=='' 
                                                                     else x[0:2]+":"+x[2:4]+":"+x[4:6])
            g['time_exit_orig_formated']=g['time_exit_orig'].apply(lambda x: np.nan if x=='' 
                                                                     else x[0:2]+":"+x[2:4]+":"+x[4:6])
            
            g['time_entry']=g.apply(lambda x: 
                                  np.nan if x['time_entry_orig']==''
                                  else
                                  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
                                  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_entry_orig_formated']
                                  , axis=1)
            
            g['change_day_entry']=g.apply(lambda x: 
                                  np.nan if x['time_entry_orig']==''
                                  else
                                  0
                                  if int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  1
                                  , axis=1)
            

            g['time_exit']=g.apply(lambda x: 
                                  np.nan if x['time_exit_orig']==''
                                  else
                                  x['date_ini'].strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
                                  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] and int(x['time_entry_orig'][0:2])>=x['hour_ini'] else
                                  (x['date_ini']+timedelta(days=1)).strftime('%Y-%m-%d')+" "+x['time_exit_orig_formated']
                                  , axis=1)
                
            
            g['change_day_exit']=g.apply(lambda x: 
                                  np.nan if x['time_exit_orig']==''
                                  else
                                  0
                                  if int(x['time_exit_orig'][0:2])>=x['hour_ini'] else
                                  1
                                  , axis=1)
          



        g=g[['trajectory_id', 'type','from_last', 'to_from_airport_id', 'geopoint_entry_id', 'geopoint_exit_id', 'fl_entry', 'fl_exit', 'distance_radius_nm', 'time_entry', 'time_exit', 'distance_entry', 'distance_exit']]
        
    else:
        g=None
        
    if read_points_on_demand:
      coord_geo_dict = {}

    return g, coord_geo_dict, max_coord_id


def dataframe_requestSpeedsProfile(data,speedProfileName,dict_trajectory):
    ####
    
    if len(data.loc[(~pd.isnull(data[speedProfileName])),:])>0:
        d=(data.apply(lambda x: zip_with_scalar_divide_point(x[speedProfileName].split(" "),x['flight_id']) if 
                                  not pd.isnull(x[speedProfileName]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','allSpeedProfile']
        
        #divide the information of the ALL_FT+ into fields

        g=f[['flight_id']].copy()
        g['trajectory_id']=g['flight_id'].apply(lambda x: dict_trajectory.get(x))    
        g['request_orig']=f['allSpeedProfile']
        g['fl']=g['request_orig'].apply(lambda x: x[0][1:len(x[0])] if x[0]!='' else np.nan)
        g['speed']=g['request_orig'].apply(lambda x: x[1][1:len(x[1])] if x[1]!='' else np.nan)
        g['speed_units']=g['request_orig'].apply(lambda x: x[1][0] if x[1]!='' else np.nan)
        g['distance']=g['request_orig'].apply(lambda x: x[2] if x[2]!='' else np.nan)

    else:
        g=None
        
    return g
        

def dataframe_eet_fir(data,dict_trajectory,sect_dict):
    ####
    eet_fir_name='ftfmEetFirList'
    
    if len(data.loc[(~pd.isnull(data[eet_fir_name])),:])>0:
        d=(data.apply(lambda x: zip_with_scalar_divide_point(x[eet_fir_name].split(" "),x['flight_id']) if 
                                  not pd.isnull(x[eet_fir_name]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','all_eet_fir']
        
        #divide the information of the ALL_FT+ into fields
        g=f[['flight_id']].copy()
        g['trajectory_id']=g['flight_id'].apply(lambda x: dict_trajectory.get(x))    
        g['eet_fir_orig']=f['all_eet_fir']
        g['fir_sid']=f['all_eet_fir'].apply(lambda x: x[0])
        g['fir_id']=g['fir_sid'].apply(lambda x: sect_dict.get(x,np.nan))
        g['eet']=f['all_eet_fir'].apply(lambda x: x[1])

        g.insert(0, 'order', (g.groupby(['flight_id']).cumcount()+1))

    else:
        g=None
        
    return g


def dataframe_eet_points(data,dict_trajectory,wpt_geo_dict,airports_geo_dict,coord_geo_dict,max_coord_id,engine):
    eet_pnt_name='ftfmEetPtList'

    read_points_on_demand = (len(coord_geo_dict)==0)#read points on demand
    
    if len(data.loc[(~pd.isnull(data[eet_pnt_name])),:])>0:
        d=(data.apply(lambda x: zip_with_scalar_divide_point(x[eet_pnt_name].split(" "),x['flight_id']) if 
                                  not pd.isnull(x[eet_pnt_name]) else np.nan, axis=1))

        d=[ x for x in d if not pd.isnull(x)]

        f=pd.DataFrame([item for sublist in [list(gen) for gen in d] for item in sublist])

        f.columns = ['flight_id','all_eet_pnt']
        
        #divide the information of the ALL_FT+ into fields
        g=f[['flight_id']].copy()
        g['trajectory_id']=g['flight_id'].apply(lambda x: dict_trajectory.get(x))    
        g['all_eet_pnt_orig']=f['all_eet_pnt']
        g['point_sid']=f['all_eet_pnt'].apply(lambda x: x[0])
        g['eet']=f['all_eet_pnt'].apply(lambda x: x[1]) 

        
        g['geopoint_id']=g['point_sid'].apply(lambda x: airports_geo_dict.get(x)['geo_id']
                             if not pd.isnull(airports_geo_dict.get(x,np.nan))
                             else wpt_geo_dict.get(x)['geo_id']
                             if not pd.isnull(wpt_geo_dict.get(x,np.nan))
                             else coord_geo_dict.get(x)['geo_id']
                             if not pd.isnull(coord_geo_dict.get(x,np.nan))
                             else np.nan)     

        #print(g[['geopoint_id','point_sid']])  

        '''
        g['geopoint_id']=g['point_sid'].apply(lambda x: airports_geo_dict.get(x)['geo_id'] 
                                              if not pd.isnull(airports_geo_dict.get(x,np.nan))
                                         else np.nan)
        g['geopoint_id']=g.apply(lambda x: wpt_geo_dict.get(x['point_sid'])['geo_id'] 
                                 if not pd.isnull(wpt_geo_dict.get(x['point_sid'],np.nan))
                                         else x['geopoint_id'], axis=1)

        g['geopoint_id']=g.apply(lambda x: coord_geo_dict.get(x['point_sid'])['geo_id'] 
                                 if not pd.isnull(coord_geo_dict.get(x['point_sid'],np.nan))
                                         else x['geopoint_id'], axis=1)
        '''

        g['missingCoord']=g['point_sid'].apply(lambda x: x if len(x)>5 else np.nan)
        missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_id']),['missingCoord']].missingCoord.unique())
        missing_geo_points.columns=['sid']

        indexes_missing=pd.isnull(g['geopoint_id'])

        missing_geo_points=missing_geo_points.loc[~pd.isnull(missing_geo_points['sid']),:]


        if read_points_on_demand:
          dict_coord_points_extra = read_coordpoints_with_geopoints(engine,sid=list(missing_geo_points['sid']))
          coord_geo_dict.update(dict_coord_points_extra)

          g['geopoint_id']=g['point_sid'].apply(lambda x: airports_geo_dict.get(x)['geo_id']
                             if not pd.isnull(airports_geo_dict.get(x,np.nan))
                             else wpt_geo_dict.get(x)['geo_id']
                             if not pd.isnull(wpt_geo_dict.get(x,np.nan))
                             else coord_geo_dict.get(x)['geo_id']
                             if not pd.isnull(coord_geo_dict.get(x,np.nan))
                             else np.nan)  

          g['missingCoord']=g['point_sid'].apply(lambda x: x if len(x)>5 else np.nan)
          missing_geo_points=pd.DataFrame(g.loc[pd.isnull(g['geopoint_id']),['missingCoord']].missingCoord.unique())
          missing_geo_points.columns=['sid']

          indexes_missing=pd.isnull(g['geopoint_id'])

          missing_geo_points=missing_geo_points.loc[~pd.isnull(missing_geo_points['sid']),:]


        
        if not missing_geo_points.empty:
            #need to add points
            max_coord_id=read_maxid_coordpoints(engine)
            
            add_missing_coordinate_geopoints_eet(engine,missing_geo_points)
            
            coord_geo_dict_extra=read_coordpoints_with_geopoints(engine,max_coord_id)
            coord_geo_dict.update(coord_geo_dict_extra)
            max_coord_id=read_maxid_coordpoints(engine)
            

            #g['geopoint_id']=g.apply(lambda x: coord_geo_dict.get(x['point_sid'])['geo_id'] 
            #                         if not pd.isnull(coord_geo_dict.get(x['point_sid'],np.nan))
            #                         else np.nan, axis=1)

            g.loc[indexes_missing,['geopoint_id']]=g.loc[indexes_missing]['point_sid']\
                                     .apply(lambda x: coord_geo_dict.get(x)['geo_id']
                                     if not pd.isnull(coord_geo_dict.get(x,np.nan))
                                     else np.nan)


        g.insert(0, 'order', (g.groupby(['flight_id']).cumcount()+1))
            
    else:
        g=None



    #print("----")
    #print(g[['geopoint_id','point_sid']]) 

    if read_points_on_demand:
      coord_geo_dict = {}
        
    return g, coord_geo_dict, max_coord_id
    


def add_missing_coordinate_geopoints_eet(engine,missing_geo_points):
    max_coord_id=read_maxid_coordpoints(engine)
    max_geo_id=read_maxid_geopoints(engine)
    
    add_coordpoints(engine,missing_geo_points)
    coord_dict=read_coordpoints(engine,max_coord_id)
    
    missing_geo_points['lat']=missing_geo_points['sid'].apply(lambda x: extract_lat(x))
    missing_geo_points['lon']=missing_geo_points['sid'].apply(lambda x: extract_lon(x))
    missing_geo_points['latlon']=missing_geo_points["lat"].map(str) +"_"+ missing_geo_points["lon"].map(str)
    missing_geo_points['type']="GEO"

    add_geopoints(engine,missing_geo_points)
    geopoints=read_geopoints_ids(engine,from_p_id=max_geo_id,rounded=1)
    
    missing_geo_points['coordpoint_id']=missing_geo_points['sid'].apply(lambda x: coord_dict.get(x))
    missing_geo_points['lat_rounded']=missing_geo_points['lat'].apply(lambda x: str(truncate(x,12)))#str(int(x * 10**12)/ 10**12))
    missing_geo_points['lon_rounded']=missing_geo_points['lon'].apply(lambda x: str(truncate(x,12)))#str(int(x * 10**12)/ 10**12))
    missing_geo_points['lat_lon']=missing_geo_points["lat_rounded"] +"_"+ missing_geo_points["lon_rounded"]
    
    missing_geo_points['geopoint_id']=missing_geo_points['lat_lon'].apply(lambda x: geopoints.get(x)['id'])
    
    add_coordpoint_point_connection(engine,missing_geo_points[['coordpoint_id','geopoint_id']])