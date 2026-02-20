const Lighting = () => (
  <>
    <ambientLight args={[{ color: [1, 1, 1], intensity: 0.6 }]} />
    <directionalLight args={[{ color: [1, 1, 1], intensity: 0.5 }]} position={[30, 30, 50]} castShadow />
  </>
)

export default Lighting
