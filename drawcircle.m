function []=drawcircle(x,y,r,clr,sy)
   
    scatter(x, y,30,clr,sy);
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    plot(xunit, yunit,clr);
end