function webm2gif() {
    #ffmpeg -y -i "$1" -vf palettegen _tmp_palette.png
    #ffmpeg -y -i "$1" -i _tmp_palette.png -filter_complex paletteuse -r 10  "${1%.webm}.gif"
    #rm _tmp_palette.png

    # create folder for temp files if not exists at /tmp/
    mkdir -p /tmp/user_ffmpeg

    ffmpeg -i "$1" -filter_complex "scale=w=960:h=-1:flags=lanczos, palettegen=stats_mode=diff" /tmp/user_ffmpeg/palette.png
    ffmpeg -i "$1" -r 0.1 -f image2 /tmp/user_ffmpeg/image_%06d.png
    ffmpeg -framerate 0.1 -i /tmp/user_ffmpeg/image_%06d.png -i /tmp/user_ffmpeg/palette.png -filter_complex "[0]scale=w=960:h=-1[x];[x][1:v] paletteuse" -pix_fmt rgb24 output.gif

}


#webm2gif "/Users/admin/Downloads/streamlit-main-2023-10-07-18-10-83.webm"

ffmpeg -ss 00:00:00 -t 35 -i /Users/admin/Downloads/streamlit-main-2023-10-07-18-10-83.webm -vf "fps=60,scale=1920:-1:flags=lanczos" -c:v gif output_full.gif
ffmpeg -i output_full.gif -filter_complex "[0:v]setpts=0.25*PTS[v]" -map "[v]" output.gif
