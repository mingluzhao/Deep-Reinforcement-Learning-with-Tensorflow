from xml.dom import minidom
import sys
import os
dirName = os.path.dirname(__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(dirName, '.. .. ..'))


def generateXML(numPredators, numPreys, numBlocks, hasWalls, dt):
    # numPredators = 3
    # numPreys = 1
    # numBlocks = 2
    # hasWalls = 2

    numWalls = 4 if hasWalls != 0 else 0

    doc = minidom.Document()

    mujoco = doc.createElement('mujoco')
    doc.appendChild(mujoco)

    option = doc.createElement('option')
    option.setAttribute('gravity', '0 0 0')
    option.setAttribute('timestep', '0.02')
    mujoco.appendChild(option)

    default = doc.createElement('default')
    mujoco.appendChild(default)
    geom = doc.createElement('geom')
    geom.setAttribute('rgba', '0 0 0 1')
    default.appendChild(geom)

    worldbody = doc.createElement('worldbody')
    mujoco.appendChild(worldbody)
    light = doc.createElement('light')
    light.setAttribute('diffuse', ".2 .2 .2")
    light.setAttribute('pos', "0 0 20")
    light.setAttribute('dir', "0 0 -1")
    light.setAttribute('mode', "track")
    worldbody.appendChild(light)

    body = doc.createElement('body')
    worldbody.appendChild(body)

    geom = doc.createElement('geom')
    geom.setAttribute('name', "floor")
    geom.setAttribute('pos', "0 0 -0.4")
    geom.setAttribute('size', "10 10 .1")
    geom.setAttribute('mass', "10000")
    geom.setAttribute('type', "box")
    geom.setAttribute('condim', "3")
    geom.setAttribute('rgba', ".9 .9 .9 1")
    body.appendChild(geom)

    camera = doc.createElement('camera')
    camera.setAttribute('name', "center")
    camera.setAttribute('mode', "fixed")
    camera.setAttribute('pos', "0 0 5")
    body.appendChild(camera)

    camera = doc.createElement('camera')
    camera.setAttribute('name', "30")
    camera.setAttribute('mode', "fixed")
    camera.setAttribute('pos', "0 -5 10")
    camera.setAttribute('axisangle', "1 0 0 30")
    body.appendChild(camera)

    for i in range(numPredators):
        body = doc.createElement('body')
        worldbody.appendChild(body)
        body.setAttribute('name', "predator" + str(i))
        body.setAttribute('pos', "0 0 0.075")

        joint = doc.createElement('joint')
        joint.setAttribute('axis', "1 0 0")
        joint.setAttribute('damping', "2.5")
        joint.setAttribute('frictionloss', "0")
        joint.setAttribute('name', "predator" + str(i) + str(0))
        joint.setAttribute('pos', "0 0 0")
        joint.setAttribute('type', "slide")
        body.appendChild(joint)

        joint = doc.createElement('joint')
        joint.setAttribute('axis', "0 1 0")
        joint.setAttribute('damping', "2.5")
        joint.setAttribute('frictionloss', "0")
        joint.setAttribute('name', "predator" + str(i) + str(1))
        joint.setAttribute('pos', "0 0 0")
        joint.setAttribute('type', "slide")
        body.appendChild(joint)

        geom = doc.createElement('geom')
        geom.setAttribute('type', "cylinder")
        geom.setAttribute('size', "0.075 0.075")
        geom.setAttribute('mass', "1")
        geom.setAttribute('conaffinity', "1")
        geom.setAttribute('solimp', "0.9 0.9999 0.001 0.5 1")
        geom.setAttribute('solref', "0.5 0.2")
        geom.setAttribute('rgba', "1 0 0 1")
        body.appendChild(geom)

        site = doc.createElement('site')
        site.setAttribute('name', "predator" + str(i))
        site.setAttribute('pos', "-0.1 0 0.05")
        site.setAttribute('type', "sphere")
        site.setAttribute('size', "0.001")
        body.appendChild(site)

    for i in range(numPreys):
        body = doc.createElement('body')
        worldbody.appendChild(body)
        body.setAttribute('name', "prey" + str(i))
        body.setAttribute('pos', "0 0 0.075")

        joint = doc.createElement('joint')
        joint.setAttribute('axis', "1 0 0")
        joint.setAttribute('damping', "2.5")
        joint.setAttribute('frictionloss', "0")
        joint.setAttribute('name', "prey" + str(i) + str(0))
        joint.setAttribute('pos', "0 0 0")
        joint.setAttribute('type', "slide")
        body.appendChild(joint)

        joint = doc.createElement('joint')
        joint.setAttribute('axis', "0 1 0")
        joint.setAttribute('damping', "2.5")
        joint.setAttribute('frictionloss', "0")
        joint.setAttribute('name', "prey" + str(i) + str(1))
        joint.setAttribute('pos', "0 0 0")
        joint.setAttribute('type', "slide")
        body.appendChild(joint)

        geom = doc.createElement('geom')
        geom.setAttribute('type', "cylinder")
        geom.setAttribute('size', "0.05 0.075")
        geom.setAttribute('mass', "1")
        geom.setAttribute('conaffinity', "1")
        geom.setAttribute('solimp', "0.9 0.9999 0.001 0.5 1")
        geom.setAttribute('solref', "0.5 0.2")
        geom.setAttribute('rgba', "0 1 0 1")
        body.appendChild(geom)

        site = doc.createElement('site')
        site.setAttribute('name', "prey" + str(i))
        site.setAttribute('pos', "0.5 0 0.075")
        site.setAttribute('type', "sphere")
        site.setAttribute('size', "0.001")
        body.appendChild(site)

    for i in range(numBlocks):
        body = doc.createElement('body')
        worldbody.appendChild(body)
        body.setAttribute('name', "block" + str(i))
        body.setAttribute('pos', "0 0 0.075")

        geom = doc.createElement('geom')
        geom.setAttribute('type', "cylinder")
        geom.setAttribute('size', "0.2 0.2")
        geom.setAttribute('mass', "1")
        geom.setAttribute('conaffinity', "1")
        geom.setAttribute('solimp', "0.9 0.9999 0.001 0.5 1")
        geom.setAttribute('solref', "0.5 0.2")
        geom.setAttribute('rgba', "0.4 0.4 0.4 0.7")
        geom.setAttribute('condim', "3")
        body.appendChild(geom)

    wallPos = hasWalls + 0.25
    wallsPos = ['{} 0 -0.2'.format(-wallPos),
                '{} 0 -0.2'.format(wallPos),
                '0 {} -0.2'.format(-wallPos),
                '0 {} -0.2'.format(wallPos)]
    wallSize = hasWalls + 0.5
    wallsSizes = ['0.25 {} 0.5'.format(wallSize),
                  '0.25 {} 0.5'.format(wallSize),
                  '{} 0.25 0.5'.format(wallSize),
                  '{} 0.25 0.5'.format(wallSize)]

    for i in range(numWalls):
        body = doc.createElement('body')
        worldbody.appendChild(body)
        geom = doc.createElement('geom')

        geom.setAttribute('name', "wall" + str(i + 1))
        geom.setAttribute('pos', wallsPos[i])
        geom.setAttribute('size', wallsSizes[i])
        geom.setAttribute('mass', "10000")
        geom.setAttribute('type', "box")
        geom.setAttribute('rgba', "0.4 0.4 0.4 0.3")
        geom.setAttribute('condim', "3")
        body.appendChild(geom)

    actuator = doc.createElement('actuator')
    mujoco.appendChild(actuator)

    for i in range(numPredators):
        motor = doc.createElement('motor')
        motor.setAttribute('gear', "1")
        motor.setAttribute('joint', "predator" + str(i) + str(0))
        actuator.appendChild(motor)

        motor = doc.createElement('motor')
        motor.setAttribute('gear', "1")
        motor.setAttribute('joint', "predator" + str(i) + str(1))
        actuator.appendChild(motor)

    for i in range(numPreys):
        motor = doc.createElement('motor')
        motor.setAttribute('gear', "1")
        motor.setAttribute('joint', "prey" + str(i) + str(0))
        actuator.appendChild(motor)

        motor = doc.createElement('motor')
        motor.setAttribute('gear', "1")
        motor.setAttribute('joint', "prey" + str(i) + str(1))
        actuator.appendChild(motor)

    root = doc.childNodes[0]
    # root.toprettyxml(encoding="utf-8")

    envPath = os.path.join(dirName, 'dt='+str(dt))
    if not os.path.exists(envPath):
        os.makedirs(envPath)

    fileName = 'hasWalls={}_numBlocks={}_numSheeps={}_numWolves={}.xml'.format(hasWalls, numBlocks, numPreys,
                                                                               numPredators)
    filePath = os.path.join(envPath, fileName)

    xml_str = root.toprettyxml()
    with open(filePath, "w") as f:
        f.write(xml_str)

generateXML(numPredators = 2, numPreys = 2, numBlocks = 2, hasWalls = 2, dt = 0.02)