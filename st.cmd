#!../../bin/linux-x86_64/experimentIOC

## You may have to change experimentIOC to something else
## everywhere it appears in this file

< envPaths

## Register all support components
dbLoadDatabase("../../dbd/experimentIOC.dbd",0,0)
experimentIOC_registerRecordDeviceDriver(pdbbase)

##### Load Devices #####
#< motor.cmd.SMCpegasus_CtTopo
#< motor.cmd.SMCpegasus_Rfa
#< smaractmcs2.iocsh_CtTopo
#< smaractmcs2.iocsh_Rfa
#< asyn.cmd.Elcomat3000
#< elettra_furnace_stream.cmd
#< eurotherm2k_modbus.cmd
#< MCBL2805.cmd
#< motor.cmd.CN30
#< motor.cmd.mmc100
#< motor.cmd.PIHexapod
#< zebra.iocsh

< webcams.cmd

## Load common IOC record instances
dbLoadRecords("$(IOCSTATS)/db/iocAdminSoft.db","IOC=iocRfa")

## Load CT record instances
dbLoadRecords("$(SUPPORT)/templates/calculateFilenamesCt.template", "exp=Ct, cam=PCOEdge")
#dbLoadRecords("$(SUPPORT)/templates/calculateFilenamesCt_PaulZaslansky.template", "exp=Ct, cam=PCOEdge")
dbLoadRecords("$(SUPPORT)/templates/micos_w_velo_as_motor.template", "exp=Ct")
dbLoadRecords("$(SUPPORT)/templates/convertMicroscopeRot.template", "motor=OMS58:25010000")
dbLoadRecords("$(SUPPORT)/templates/CTFloatPixelsize.template")
#dbLoadRecords("$(SUPPORT)/templates/PCOEdge_pseudo_motors.template")

## Load RFA record instances
dbLoadRecords("$(SUPPORT)/templates/monitorMotorRBVDiff.template", "name=DMMTheta, motor1=OMS58:25000007, motor2=OMS58:25001003")
dbLoadRecords("$(SUPPORT)/templates/mbbo15Strings.template", "exp=Rfa, name=Info")
dbLoadRecords("$(SUPPORT)/templates/k428_fanout.template", "P=K428:all:")
dbLoadRecords("$(SUPPORT)/templates/waveform.template", "exp=Rfa, name=LongString")

## Load TOPO record instances
dbLoadRecords("$(SUPPORT)/templates/calculateFilenamesTopo.template", "exp=Topo, cam=XRFDS, cam2=Manta, format=TIFF")
dbLoadRecords("$(SUPPORT)/templates/DCMController.template")
dbLoadRecords("$(SUPPORT)/templates/TOPOController.template")

## Autosave restore
set_savefile_path("/soft/autosave/Experiment")
set_requestfile_path("/soft/autosave/Experiment")
set_requestfile_path("/soft/autosave/reqTemplates")
set_requestfile_path("$(ADCORE)/ADApp/Db")
set_requestfile_path("$(ADCORE)/iocBoot")
set_requestfile_path("$(ADURL)/urlApp/Db")
set_pass0_restoreFile("Ct.sav")
set_pass0_restoreFile("Rfa.sav")
set_pass0_restoreFile("topo.sav")
set_pass0_restoreFile("webcams.sav")
#set_pass0_restoreFile("eurotherm2k.sav")
#set_pass0_restoreFile("PEGAS0101.sav")
#set_pass0_restoreFile("justHlmLlm.sav")
set_pass1_restoreFile("Ct.sav")
set_pass1_restoreFile("Rfa.sav")
set_pass1_restoreFile("topo.sav")
set_pass1_restoreFile("webcams.sav")
#set_pass1_restoreFile("eurotherm2k.sav")
#set_pass1_restoreFile("PEGAS0101.sav")
#set_pass1_restoreFile("justHlmLlm.sav")
save_restoreSet_status_prefix("experiment:")
save_restoreSet_DatedBackupFiles(1)
save_restoreSet_NumSeqFiles(1)
## Time interval between sequenced backups
save_restoreSet_SeqPeriodInSeconds(600)

iocInit
epicsThreadSleep(3.)

## motorUtil (allstop & alldone)
#motorUtilInit("PIF206:")
#motorUtilInit("smarAct:")
#motorUtilInit("PiCo33:")
#motorUtilInit("faulhaber:")
#motorUtilInit("micronix:")
#motorUtilInit("PEGAS:miocb0101")
#motorUtilInit("PEGAS:miocb0102")


## Autosave requests
create_monitor_set("Ct.req", 5,"NULL")
create_monitor_set("Rfa.req", 5)
create_monitor_set("topo.req", 5)
create_monitor_set("webcams.req", 30, "P=$(PREFIX), P2=$(PREFIX2), P3=$(PREFIX3), P4=$(PREFIX4), P5=$(PREFIX5)")
#create_monitor_set("eurotherm2k.req", 30, "P=EUROTHERM, Q=2K")
#create_monitor_set("PEGAS0101.req", 30)
#create_monitor_set("justHlmLlm.req", 30, "P=smarAct:, N=1")

# sometimes the SP is not processed to the hardware so process it every 10s
#dbpf "EUROTHERM2K:SP.SCAN","3"

# Hexapod closedLoop is enabled
#dbpf "PIF206:x.CNEN","1"
#dbpf "PIF206:y.CNEN","1"
#dbpf "PIF206:z.CNEN","1"
#dbpf "PIF206:rotx.CNEN","1"
#dbpf "PIF206:roty.CNEN","1"
#dbpf "PIF206:rotz.CNEN","1"

# 0.5mm is a realistic retry-deadband for the JfA-Schlitten
#dbpf "faulhaber:m1.RDBD","0.5"
# 3 seconds delay for the motor to reckon a movement as completed
#dbpf "faulhaber:m1.DLY","3"
