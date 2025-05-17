window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	// Check for click events on the navbar burger icon

	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	bulmaSlider.attach();

})

function changeVideo() {
	const selector = document.getElementById('video-selector');
	const video = document.getElementById('selected-video');
	const source = video.querySelector('source');

	// Update the video source
	source.src = selector.value;

	// Reload the video to apply the new source
	video.load();
	video.play();
}
