package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/golang/geo/r3"
	dem "github.com/markus-wa/demoinfocs-golang"
	"github.com/markus-wa/demoinfocs-golang/common"
	"github.com/markus-wa/demoinfocs-golang/events"
	"github.com/mholt/archiver"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type AlivePlayers struct {
	Name        string                  `json:"Name"`
	Side        string                  `json:"Side"`
	Position    []float64               `json:"Position"`
	Weapon      common.EquipmentElement `json:"Weapon"`
	WeaponValue int                     `json:"WeaponValue"`
	Money       int                     `json:"Money"`
	Velocity    []float64               `json:"Velocity"`
	Kills       int                     `json:"Kills"`
}

type PostPlantStatus struct {
	Second       time.Duration  `json:"second"`
	TsideAlive   int            `json:"TsideAlive"`
	CTsideAlive  int            `json:"CTsideAlive"`
	AlivePlayers []AlivePlayers `json:"alivePlayers"`
	TFlashBangs  int
	CTFlashBangs int
	TSmokes      int
	CTSmokes     int
	TMolo        int
	CTMolo       int
}

type RoundInfo struct {
	ParseStartTime  int64
	MapName         string            `json:"mapName"`
	RoundNum        int               `json:"num"`
	WinnerSide      string            `json:"winnerSide"`
	BombPosition    []float64         `json:"bombPosition"`
	PostPlantStatus []PostPlantStatus `json:"postPlantStatus"`
}

func newSession() (*session.Session, error) {
	ak := "BQ40V7KXWOUB1OLQ543C"
	sk := "ztM2Wg9TefNBqsAOYrXKNjfMyn2sMBhbnU6eFenv"
	creds := credentials.NewStaticCredentials(ak, sk, "")
	config := &aws.Config{
		Region:           aws.String("eu-central-1"),
		Endpoint:         aws.String("https://s3.eu-central-1.wasabisys.com"),
		S3ForcePathStyle: aws.Bool(true),
		Credentials:      creds,
		DisableSSL:       aws.Bool(false),
	}
	return session.NewSession(config)
}

const bucket = "historical_data"

func listObjects() (*s3.ListObjectsOutput, *s3.S3) {
	sess, err := newSession()

	if err != nil {
		fmt.Println("failed to create session,", err)
	}

	svc := s3.New(sess)
	listParams := &s3.ListObjectsInput{
		Bucket: aws.String(bucket), // Required
	}
	resp, err := svc.ListObjects(listParams)
	return resp, svc
}

func removeUsedFile(rarPath string, demoFolderPath string) {
	os.Remove(rarPath)
	os.RemoveAll("E:/Thesis/demoFile/extract_demos")
}

func getParsedFiles() []string{
	return []string{
		1:"demos/1000_42888", 2:"demos/1003_44939", 3:"demos/10072_48231", 4:"demos/1007_45217", 5:"demos/10098_47626",
		6:"demos/1015_43545", 7:"demos/10174_46939", 8:"demos/10188_47486", 9:"demos/1020_42164", 10:"demos/1027_47212",
		11:"demos/1028_42391", 12:"demos/1029_46104", 13:"demos/1033_45181", 14:"demos/1035_42261",
		15:"demos/1042_42654",16:"demos/1042_43698",17:"demos/1047_42953",18:"demos/1051_48312",19:"demos/1058_41993",
		20:"demos/1061_42528",21:"demos/1062_42668", 22:"demos/1066_43900", 23:"demos/10000_45178",24:"demos/1000_44064",
		25:"demos/1008_42931",26:"demos/1007_45217",27:"demos/1009_45778",28:"demos/1010_45541",29:"demos/1012_43750",
		30:"demos/1013_44196",31:"demos/1016_43547",32:"demos/10181_46750",33:"demos/10190_48084",34:"demos/1020_44868",
		35:"demos/1024_43215",36:"demos/1034_42464",37:"demos/1035_43651",38:"demos/10380_46673",39:"demos/1038_46133",
		40:"demos/1039_43904",41:"demos/10415_48105",42:"demos/1041_43775",43:"demos/10452_48637",44:"demos/10460_47106",
		45:"demos/1049_48157",46:"demos/1057_42176",47:"demos/1060_42139",48:"demos/1063_44192",49:"demos/1066_45466",
		50:"demos/10696_47101",51:"demos/1069_43281",52:"demos/1071_43613",53:"demos/1071_45549",64:"demos/1072_45943",
		65:"demos/1256_45924", 66:"demos/1264_44529",
	}
}

func ifParsed(parsed []string, key string) bool{
	for _,id := range parsed{
		if key == id{
			return true
		}
	}

	return false
}
func main() {
	roundInfos := make([]RoundInfo, 0)
	alivePlayers := make([]AlivePlayers, 0)
	postPlantStatuses := make([]PostPlantStatus, 0)
	gameDatas := make([][]RoundInfo, 0)
	var demoFolderPath string
	mirageNum:=0
	parsed :=getParsedFiles()

	resp, svc := listObjects()
	for _, value := range resp.Contents {
		key := *value.Key
		if ifParsed(parsed, key){
			continue
		}
		inputParams := &s3.GetObjectInput{
			Bucket: aws.String(bucket), // Required
			Key:    aws.String(key),    // Required
		}
		input, err := svc.GetObject(inputParams)

		rarPath := "E:/Thesis/demoFile/test_demo.rar"
		rarFile, err := os.Create(rarPath)
		if err != nil {
			fmt.Println("failed to create file,", err)
		}

		io.Copy(rarFile, input.Body)

		demoFolderPath = "E:/Thesis/demoFile/extract_" + key
		r := archiver.NewRar()
		err = r.Unarchive(rarPath, demoFolderPath)
		if err != nil {
			fmt.Printf("ERROR!\n")
			parsed = append(parsed,key)
			continue
		}
		filepath.Walk(demoFolderPath,
			func(path string, info os.FileInfo, err error) error {
				if err != nil {
					fmt.Printf("ERROR!\n")
				}
				if info.IsDir() != true {
					fmt.Printf("Input file found \n")
					fmt.Printf("%s \n ", path)
					parsed = append(parsed,key)
					if strings.Contains(path,"mirage"){
						mirageNum++
						if mirageNum > 59{
							gameData := getGameData(path, roundInfos, alivePlayers, postPlantStatuses)
							gameDatas = append(gameDatas, gameData)
							fmt.Printf("Done parsing file \n")
						}
					}
				}
				return nil
			})
		removeUsedFile(rarPath, demoFolderPath)
		fmt.Printf("File removed \n")

		for i := 1; i <= 25; i++ {
			if len(gameDatas) == i*10{
				fileOutput(gameDatas, len(gameDatas))
				fmt.Printf("parsed: %v", parsed)
				break
			}

		}
	}

	fileOutput(gameDatas, len(gameDatas))

}

func getGameData(demoPath string, roundInfos []RoundInfo, alivePlayers []AlivePlayers,
	postPlantStatuses []PostPlantStatus) []RoundInfo {
	f, err := os.Open(demoPath)
	defer f.Close()
	p := dem.NewParser(f)
	header, err := p.ParseHeader()
	mapName := header.MapName
	parseTime := time.Now().Unix()

	if mapName == "de_mirage" {
		// rounds -> time-since-plant -> player-name -> position
		postPlantPositions := make(map[int]map[time.Duration]map[string]r3.Vector)
		postPlantWeapons := make(map[int]map[time.Duration]map[string]common.EquipmentElement)
		postPlantMoney := make(map[int]map[time.Duration]map[string]int)
		postPlantCTAlives := make(map[int]map[time.Duration]int)
		postPlantTAlives := make(map[int]map[time.Duration]int)
		postPlantVelocity := make(map[int]map[time.Duration]map[string]r3.Vector)
		postPlantEquipmentValue := make(map[int]map[time.Duration]map[string]int)
		postPlantTeam := make(map[int]map[time.Duration]map[string]string)
		postPlantKills := make(map[int]map[time.Duration]map[string]int)
		playerNames := make(map[int]map[time.Duration]map[string]string)
		roundWinner := make(map[int]string)
		postPlantFlashBangs := make(map[int]map[time.Duration]map[string]int)
		postPlantSmokes := make(map[int]map[time.Duration]map[string]int)
		postPlantMolo := make(map[int]map[time.Duration]map[string]int)

		bombPositions := make(map[int]r3.Vector)

		var currentPPPos map[time.Duration]map[string]r3.Vector
		var currentWeapon map[time.Duration]map[string]common.EquipmentElement
		var currentMoney map[time.Duration]map[string]int
		var currentCTAlive map[time.Duration]int
		var currentTAlive map[time.Duration]int
		var currentVelocity map[time.Duration]map[string]r3.Vector
		var currentEquipmentValue map[time.Duration]map[string]int
		var currentTeam map[time.Duration]map[string]string
		var currentKills map[time.Duration]map[string]int
		var currentPlayerNames map[time.Duration]map[string]string
		var lastPlantTime time.Duration
		var lastSnapshotTime time.Duration
		var isBombPlantActive bool
		var bombPosition r3.Vector
		var currentFlashBangs map[time.Duration]map[string]int
		var currentSmokes map[time.Duration]map[string]int
		var currentMolo map[time.Duration]map[string]int
		

		// snapshot when the bomb gets planted
		p.RegisterEventHandler(func(e events.BombPlanted) {
			isBombPlantActive = true
			lastPlantTime = p.CurrentTime()
			lastSnapshotTime = p.CurrentTime()

			bombPosition = p.GameState().Bomb().Position()

			currentTeam = make(map[time.Duration]map[string]string)
			currentPlayerNames = make(map[time.Duration]map[string]string)
			currentPPPos = make(map[time.Duration]map[string]r3.Vector)
			currentWeapon = make(map[time.Duration]map[string]common.EquipmentElement)
			currentMoney = make(map[time.Duration]map[string]int)
			currentVelocity = make(map[time.Duration]map[string]r3.Vector)
			currentEquipmentValue = make(map[time.Duration]map[string]int)
			currentKills = make(map[time.Duration]map[string]int)
			currentCTAlive = make(map[time.Duration]int)
			currentTAlive = make(map[time.Duration]int)
			currentFlashBangs = make(map[time.Duration]map[string]int)
			currentSmokes = make(map[time.Duration]map[string]int)
			currentMolo = make(map[time.Duration]map[string]int)

			currentPPPos[0], currentWeapon[0], currentMoney[0], currentVelocity[0],
				currentEquipmentValue[0], currentTeam[0], currentKills[0],
				currentPlayerNames[0], currentTAlive[0], currentCTAlive[0],
				currentFlashBangs[0], currentSmokes[0], currentMolo[0] = Snapshot(p)

		})

		// snapshot positions every 5 seconds
		p.RegisterEventHandler(func(e events.FrameDone) {
			const snapshotFrequency = 5 * time.Second

			now := p.CurrentTime()
			if isBombPlantActive && (lastSnapshotTime+snapshotFrequency) < now {
				lastSnapshotTime = now
				currentPPPos[now-lastPlantTime], currentWeapon[now-lastPlantTime], currentMoney[now-lastPlantTime],
					currentVelocity[now-lastPlantTime], currentEquipmentValue[now-lastPlantTime],
					currentTeam[now-lastPlantTime], currentKills[now-lastPlantTime],
					currentPlayerNames[now-lastPlantTime], currentTAlive[now-lastPlantTime],
					currentCTAlive[now-lastPlantTime], currentFlashBangs[now-lastPlantTime], currentSmokes[now-lastPlantTime],
					currentMolo[now-lastPlantTime] = Snapshot(p)
				bombPosition = p.GameState().Bomb().Position()
			}
		})

		// store post-plant positions at the end of the round
		p.RegisterEventHandler(func(e events.RoundEnd) {
			if !isBombPlantActive {
				return
			}
			isBombPlantActive = false
			postPlantPositions[p.GameState().TotalRoundsPlayed()] = currentPPPos
			bombPositions[p.GameState().TotalRoundsPlayed()] = bombPosition
			postPlantCTAlives[p.GameState().TotalRoundsPlayed()] = currentCTAlive
			postPlantTAlives[p.GameState().TotalRoundsPlayed()] = currentTAlive
			postPlantWeapons[p.GameState().TotalRoundsPlayed()] = currentWeapon
			postPlantMoney[p.GameState().TotalRoundsPlayed()] = currentMoney
			postPlantVelocity[p.GameState().TotalRoundsPlayed()] = currentVelocity
			postPlantEquipmentValue[p.GameState().TotalRoundsPlayed()] = currentEquipmentValue
			postPlantTeam[p.GameState().TotalRoundsPlayed()] = currentTeam
			postPlantKills[p.GameState().TotalRoundsPlayed()] = currentKills
			playerNames[p.GameState().TotalRoundsPlayed()] = currentPlayerNames
			if e.WinnerState.Team() == 2 {
				roundWinner[p.GameState().TotalRoundsPlayed()] = "T"
			}
			if e.WinnerState.Team() == 3 {
				roundWinner[p.GameState().TotalRoundsPlayed()] = "CT"
			}
			postPlantFlashBangs[p.GameState().TotalRoundsPlayed()] = currentFlashBangs
			postPlantSmokes[p.GameState().TotalRoundsPlayed()] = currentSmokes
			postPlantMolo[p.GameState().TotalRoundsPlayed()] = currentMolo

		})
		// Parse to end
		err = p.ParseToEnd()
		checkError(err)

		// just to make sure, maybe we didn't get a RoundEnd event for the final round
		if isBombPlantActive {
			postPlantPositions[p.GameState().TotalRoundsPlayed()] = currentPPPos
			bombPositions[p.GameState().TotalRoundsPlayed()] = bombPosition
			postPlantCTAlives[p.GameState().TotalRoundsPlayed()] = currentCTAlive
			postPlantTAlives[p.GameState().TotalRoundsPlayed()] = currentTAlive
			postPlantWeapons[p.GameState().TotalRoundsPlayed()] = currentWeapon
			postPlantMoney[p.GameState().TotalRoundsPlayed()] = currentMoney
			postPlantVelocity[p.GameState().TotalRoundsPlayed()] = currentVelocity
			postPlantEquipmentValue[p.GameState().TotalRoundsPlayed()] = currentEquipmentValue
			postPlantTeam[p.GameState().TotalRoundsPlayed()] = currentTeam
			postPlantKills[p.GameState().TotalRoundsPlayed()] = currentKills
			playerNames[p.GameState().TotalRoundsPlayed()] = currentPlayerNames
			postPlantFlashBangs[p.GameState().TotalRoundsPlayed()] = currentFlashBangs
			postPlantSmokes[p.GameState().TotalRoundsPlayed()] = currentSmokes
			postPlantMolo[p.GameState().TotalRoundsPlayed()] = currentMolo
		}

		// sort rounds, otherwise output order is random
		rounds := make([]int, 0)
		for k, _ := range playerNames {
			rounds = append(rounds, k)
		}
		sort.Ints(rounds)

		for _, roundNr := range rounds {
			ppos := postPlantPositions[roundNr]
			ppta := postPlantTAlives[roundNr]
			ppcta := postPlantCTAlives[roundNr]
			ppw := postPlantWeapons[roundNr]
			ppm := postPlantMoney[roundNr]
			ppv := postPlantVelocity[roundNr]
			ppev := postPlantEquipmentValue[roundNr]
			ppt := postPlantTeam[roundNr]
			ppk := postPlantKills[roundNr]
			ppn := playerNames[roundNr]
			rw := roundWinner[roundNr]
			ppFlash := postPlantFlashBangs[roundNr]
			ppSmoke := postPlantSmokes[roundNr]
			ppMolo := postPlantMolo[roundNr]

			snapshotTimes := make([]int, 0)
			for t, _ := range ppos {
				snapshotTimes = append(snapshotTimes, int(t))
			}
			sort.Ints(snapshotTimes)

			for _, t := range snapshotTimes {
				timeSincePlant := time.Duration(t)
				positions := ppos[timeSincePlant]
				tAlives := ppta[timeSincePlant]
				ctAlives := ppcta[timeSincePlant]
				weapons := ppw[timeSincePlant]
				money := ppm[timeSincePlant]
				velocities := ppv[timeSincePlant]
				equipmentvalues := ppev[timeSincePlant]
				kills := ppk[timeSincePlant]
				names := ppn[timeSincePlant]
				teams := ppt[timeSincePlant]
				flashs := ppFlash[timeSincePlant]
				smokes := ppSmoke[timeSincePlant]
				molos := ppMolo[timeSincePlant]

				for name := range names {
					alivePlayer := AlivePlayers{
						Name:        name,
						Side:        teams[name],
						Position:    []float64{positions[name].X, positions[name].Y, positions[name].Z},
						Weapon:      weapons[name],
						WeaponValue: equipmentvalues[name],
						Money:       money[name],
						Velocity:    []float64{velocities[name].X, velocities[name].Y, velocities[name].Z},
						Kills:       kills[name],
					}
					alivePlayers = append(alivePlayers, alivePlayer)
				}
				postPlantStatus := PostPlantStatus{
					Second:       timeSincePlant,
					TsideAlive:   tAlives,
					CTsideAlive:  ctAlives,
					AlivePlayers: alivePlayers,
					CTFlashBangs: flashs["CT"],
					TFlashBangs:  flashs["T"],
					CTSmokes:     smokes["CT"],
					TSmokes:      smokes["T"],
					CTMolo:       molos["CT"],
					TMolo:        molos["T"],
				}
				postPlantStatuses = append(postPlantStatuses, postPlantStatus)
				alivePlayers = make([]AlivePlayers, 0)
			}
			roundInfo := RoundInfo{
				ParseStartTime:  parseTime,
				MapName:         mapName,
				RoundNum:        roundNr + 1,
				WinnerSide:      rw,
				BombPosition:    []float64{bombPositions[roundNr].X, bombPositions[roundNr].Y, bombPositions[roundNr].Z},
				PostPlantStatus: postPlantStatuses,
			}
			roundInfos = append(roundInfos, roundInfo)
			postPlantStatuses = make([]PostPlantStatus, 0)
		}
		return roundInfos
	}
	return nil
}

func fileOutput(data [][]RoundInfo, demoNum int) {
	roundData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("%v\n", err)
	}
	filename := "C:/Users/admin/Desktop/Output" + strconv.Itoa(demoNum) + "v3.json"
	var file *os.File

	if checkFileIsExist(filename) {
		file, err = os.OpenFile(filename, os.O_RDWR, 0666)
		fmt.Println("Output file exists.\n")
	} else {
		file, err = os.Create(filename)
		fmt.Println("Output file created.\n")
	}

	if err != nil {
		fmt.Println("open file failed, err: ", err)
	}

	w := bufio.NewWriter(file)
	w.WriteString(string(roundData))
	w.Flush()

}
func checkFileIsExist(filename string) bool {
	var exist = true
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		exist = false
	}
	return exist
}

func Snapshot(parser *dem.Parser) (map[string]r3.Vector, map[string]common.EquipmentElement,
	map[string]int, map[string]r3.Vector, map[string]int, map[string]string, map[string]int,
	map[string]string, int, int, map[string]int, map[string]int, map[string]int) {

	position := make(map[string]r3.Vector)
	weapon := make(map[string]common.EquipmentElement)
	money := make(map[string]int)
	velocity := make(map[string]r3.Vector)
	currentEquipmentValue := make(map[string]int)
	team := make(map[string]string)
	kills := make(map[string]int)
	names := make(map[string]string)
	tAliveNum := 0
	ctAliveNum := 0
	flashBangs := make(map[string]int)
	smokes := make(map[string]int)
	molo := make(map[string]int)

	for _, pl := range parser.GameState().Participants().Playing() {
		if pl.IsAlive() == true {
			position[pl.Name] = pl.Position
			weapon[pl.Name] = pl.ActiveWeapon().Weapon
			money[pl.Name] = pl.Money
			velocity[pl.Name] = pl.Velocity
			currentEquipmentValue[pl.Name] = pl.CurrentEquipmentValue
			kills[pl.Name] = pl.AdditionalPlayerInformation.Kills
			names[pl.Name] = pl.Name
			if pl.Team == 2 {
				team[pl.Name] = "T"
				tAliveNum++
			}
			if pl.Team == 3 {
				team[pl.Name] = "CT"
				ctAliveNum++
			}
			weapons := pl.RawWeapons
			names := common.EquipmentElementNames()
			for _, i := range weapons {
				if names[i.Weapon] == "Flashbang" {
					flashBangs[team[pl.Name]]++
				}
				if names[i.Weapon] == "Smoke Grenade" {
					smokes[team[pl.Name]]++
				}
				if names[i.Weapon] == "Molotov" {
					molo[team[pl.Name]]++
				}

			}
		}
	}

	return position, weapon, money, velocity, currentEquipmentValue, team, kills, names, tAliveNum, ctAliveNum,
		flashBangs, smokes, molo
}

func getWeaponName(element common.EquipmentElement) string {
	names := common.EquipmentElementNames()
	return names[element]
}
func checkError(err error) {
	if err != nil {
		panic(err)
	}
}


