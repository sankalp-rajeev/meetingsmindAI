// Video Player Component
// TODO: Phase D - Custom controls + time sync

function VideoPlayer({ src }) {
    return (
        <div className="video-player">
            <video src={src} controls />
        </div>
    )
}

export default VideoPlayer
