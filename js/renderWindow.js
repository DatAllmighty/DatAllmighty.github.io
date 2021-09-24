import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls'
import { PLYLoader } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/loaders/PLYLoader'
import Stats from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/stats.module'

const scene = new THREE.Scene()
scene.add(new THREE.AxesHelper(5))

const light = new THREE.PointLight()
light.position.set(0, 0, 0)
scene.add(light)

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
)
camera.position.z = 0.25

const renderer = new THREE.WebGLRenderer()
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(window.innerWidth, window.innerHeight)
document.body.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true

const material = new THREE.MeshPhysicalMaterial({
    color: 0xb2ffc8,
    metalness: 0,
    roughness: 1,
    transparent: false,
    transmission: 0.0,
    side: THREE.DoubleSide,
    clearcoat: 1.0,
    clearcoatRoughness: 0.25,
    vertexColors: THREE.VertexColors 
})

const loader = new PLYLoader()


window.addEventListener('resize', onWindowResize, false)

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth/2, window.innerHeight/2)
    render()
}

const stats = Stats()
document.body.appendChild(stats.dom)

function animate() {
    requestAnimationFrame(animate)

    controls.update()

    render()
    stats.update()
}

function clearScene() {
    const object = scene.getObjectByName( 'mesh' )
    if(object) {
        object.geometry.dispose()
        object.material.dispose()
        scene.remove( object )
    }
}

function loadRoom() {
    clearScene()
    loader.load(
    'models/room.ply',
    function (geometry) {
        geometry.computeVertexNormals()
        const mesh = new THREE.Mesh(geometry, material)
        mesh.rotateX(Math.PI)
        mesh.name = 'mesh'
        scene.add(mesh)
    },
    (error) => {
        console.log(error)
    }
    )
}

function loadCurvature() {
    clearScene()
    loader.load(
    'models/roomCurvature.ply',
    function (geometry) {
        geometry.computeVertexNormals()
        const mesh = new THREE.Mesh(geometry, material)
        mesh.rotateX(Math.PI)
        mesh.name = 'mesh'
        scene.add(mesh)
    },
    (error) => {
        console.log(error)
    }
    )
}

function render() {
    renderer.render(scene, camera)
}

loadRoom()
animate()